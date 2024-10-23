// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Simple iterator over batches for use in testing

use std::fmt;
use std::{
    any::Any,
    pin::Pin,
    sync::{Arc, Weak},
    task::{Context, Poll},
};

use crate::stream::{RecordBatchReceiverStream, RecordBatchStreamAdapter};
use crate::{
    common, DisplayAs, DisplayFormatType, ExecutionMode, ExecutionPlan, Partitioning,
    PlanProperties, RecordBatchStream, SendableRecordBatchStream, Statistics,
};

use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use datafusion_common::project_schema;
use datafusion_common::{internal_err, DataFusionError, Result};
use datafusion_execution::memory_pool::MemoryReservation;
use datafusion_execution::TaskContext;
use datafusion_physical_expr::equivalence::ProjectionMapping;
use datafusion_physical_expr::expressions::Column;
use datafusion_physical_expr::utils::collect_columns;
use datafusion_physical_expr::LexOrdering;
use datafusion_physical_expr::{EquivalenceProperties, PhysicalSortExpr};

use futures::Stream;
use tokio::sync::Barrier;

// Execution plan for reading in-memory batches of data
pub struct EndlessMemoryExec {
    /// The partitions to query
    partitions: Vec<Vec<RecordBatch>>,
    single_data: RecordBatch,
    /// Schema representing the data before projection
    schema: SchemaRef,
    /// Schema representing the data after the optional projection is applied
    projected_schema: SchemaRef,
    /// Optional projection
    projection: Option<Vec<usize>>,
    // Sort information: one or more equivalent orderings
    sort_information: Vec<LexOrdering>,
    cache: PlanProperties,
    /// if partition sizes should be displayed
    show_sizes: bool,
}

impl fmt::Debug for EndlessMemoryExec {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("EndlessMemoryExec")
            .field("partitions", &"[...]")
            .field("schema", &self.schema)
            .field("projection", &self.projection)
            .field("sort_information", &self.sort_information)
            .finish()
    }
}

impl DisplayAs for EndlessMemoryExec {
    fn fmt_as(
        &self,
        t: DisplayFormatType,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                let partition_sizes: Vec<_> =
                    self.partitions.iter().map(|b| b.len()).collect();

                let output_ordering = self
                    .sort_information
                    .first()
                    .map(|output_ordering| {
                        format!(
                            ", output_ordering={}",
                            PhysicalSortExpr::format_list(output_ordering)
                        )
                    })
                    .unwrap_or_default();

                if self.show_sizes {
                    write!(
                        f,
                        "EndlessMemoryExec: partitions={}, partition_sizes={partition_sizes:?}{output_ordering}",
                        partition_sizes.len(),
                    )
                } else {
                    write!(f, "EndlessMemoryExec: partitions={}", partition_sizes.len(),)
                }
            }
        }
    }
}

impl ExecutionPlan for EndlessMemoryExec {
    fn name(&self) -> &'static str {
        "EndlessMemoryExec"
    }

    /// Return a reference to Any that can be used for downcasting
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn properties(&self) -> &PlanProperties {
        &self.cache
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        // this is a leaf node and has no children
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        // EndlessMemoryExec has no children
        if children.is_empty() {
            Ok(self)
        } else {
            internal_err!("Children cannot be replaced in {self:?}")
        }
    }

    fn execute(
        &self,
        partition: usize,
        _context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        Ok(Box::pin(EndlessMemoryStream::try_new(
            self.partitions[partition].clone(),
            self.single_data.clone(),
            Arc::clone(&self.projected_schema),
            self.projection.clone(),
        )?))
    }

    /// We recompute the statistics dynamically from the arrow metadata as it is pretty cheap to do so
    fn statistics(&self) -> Result<Statistics> {
        Ok(common::compute_record_batch_statistics(
            &self.partitions,
            &self.schema,
            self.projection.clone(),
        ))
    }
}

impl EndlessMemoryExec {
    /// Create a new execution plan for reading in-memory record batches
    /// The provided `schema` should not have the projection applied.
    pub fn try_new(
        partitions: &[Vec<RecordBatch>],
        single_data: RecordBatch,
        schema: SchemaRef,
        projection: Option<Vec<usize>>,
    ) -> Result<Self> {
        let projected_schema = project_schema(&schema, projection.as_ref())?;
        let cache =
            Self::compute_properties(Arc::clone(&projected_schema), &[], partitions);
        Ok(Self {
            partitions: partitions.to_vec(),
            single_data,
            schema,
            projected_schema,
            projection,
            sort_information: vec![],
            cache,
            show_sizes: true,
        })
    }

    /// set `show_sizes` to determine whether to display partition sizes
    pub fn with_show_sizes(mut self, show_sizes: bool) -> Self {
        self.show_sizes = show_sizes;
        self
    }

    pub fn partitions(&self) -> &[Vec<RecordBatch>] {
        &self.partitions
    }

    pub fn projection(&self) -> &Option<Vec<usize>> {
        &self.projection
    }

    /// A memory table can be ordered by multiple expressions simultaneously.
    /// [`EquivalenceProperties`] keeps track of expressions that describe the
    /// global ordering of the schema. These columns are not necessarily same; e.g.
    /// ```text
    /// ┌-------┐
    /// | a | b |
    /// |---|---|
    /// | 1 | 9 |
    /// | 2 | 8 |
    /// | 3 | 7 |
    /// | 5 | 5 |
    /// └---┴---┘
    /// ```
    /// where both `a ASC` and `b DESC` can describe the table ordering. With
    /// [`EquivalenceProperties`], we can keep track of these equivalences
    /// and treat `a ASC` and `b DESC` as the same ordering requirement.
    ///
    /// Note that if there is an internal projection, that projection will be
    /// also applied to the given `sort_information`.
    pub fn try_with_sort_information(
        mut self,
        mut sort_information: Vec<LexOrdering>,
    ) -> Result<Self> {
        // All sort expressions must refer to the original schema
        let fields = self.schema.fields();
        let ambiguous_column = sort_information
            .iter()
            .flatten()
            .flat_map(|expr| collect_columns(&expr.expr))
            .find(|col| {
                fields
                    .get(col.index())
                    .map(|field| field.name() != col.name())
                    .unwrap_or(true)
            });
        if let Some(col) = ambiguous_column {
            return internal_err!(
                "Column {:?} is not found in the original schema of the EndlessMemoryExec",
                col
            );
        }

        // If there is a projection on the source, we also need to project orderings
        if let Some(projection) = &self.projection {
            let base_eqp = EquivalenceProperties::new_with_orderings(
                self.original_schema(),
                &sort_information,
            );
            let proj_exprs = projection
                .iter()
                .map(|idx| {
                    let base_schema = self.original_schema();
                    let name = base_schema.field(*idx).name();
                    (Arc::new(Column::new(name, *idx)) as _, name.to_string())
                })
                .collect::<Vec<_>>();
            let projection_mapping =
                ProjectionMapping::try_new(&proj_exprs, &self.original_schema())?;
            sort_information = base_eqp
                .project(&projection_mapping, self.schema())
                .oeq_class
                .orderings;
        }

        self.sort_information = sort_information;
        // We need to update equivalence properties when updating sort information.
        let eq_properties = EquivalenceProperties::new_with_orderings(
            self.schema(),
            &self.sort_information,
        );
        self.cache = self.cache.with_eq_properties(eq_properties);

        Ok(self)
    }

    pub fn original_schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    /// This function creates the cache object that stores the plan properties such as schema, equivalence properties, ordering, partitioning, etc.
    fn compute_properties(
        schema: SchemaRef,
        orderings: &[LexOrdering],
        partitions: &[Vec<RecordBatch>],
    ) -> PlanProperties {
        let eq_properties = EquivalenceProperties::new_with_orderings(schema, orderings);
        PlanProperties::new(
            eq_properties,                                       // Equivalence Properties
            Partitioning::UnknownPartitioning(partitions.len()), // Output Partitioning
            ExecutionMode::Bounded,                              // Execution Mode
        )
    }
}

/// Iterator over batches
pub struct EndlessMemoryStream {
    /// Vector of record batches
    data: Vec<RecordBatch>,
    single_data: RecordBatch,
    /// Optional memory reservation bound to the data, freed on drop
    reservation: Option<MemoryReservation>,
    /// Schema representing the data
    schema: SchemaRef,
    /// Optional projection for which columns to load
    projection: Option<Vec<usize>>,
    /// Index into the data
    index: usize,
}

impl EndlessMemoryStream {
    /// Create an iterator for a vector of record batches
    pub fn try_new(
        data: Vec<RecordBatch>,
        single_data: RecordBatch,
        schema: SchemaRef,
        projection: Option<Vec<usize>>,
    ) -> Result<Self> {
        Ok(Self {
            data,
            single_data,
            reservation: None,
            schema,
            projection,
            index: 0,
        })
    }

    /// Set the memory reservation for the data
    pub(super) fn with_reservation(mut self, reservation: MemoryReservation) -> Self {
        self.reservation = Some(reservation);
        self
    }
}

impl Stream for EndlessMemoryStream {
    type Item = Result<RecordBatch>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        _: &mut Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        // Poll::Ready(Some(Ok(self.single_data.clone())))

        Poll::Ready(if self.index < self.data.len() {
            self.index += 1;
            let batch = &self.data[self.index - 1];

            // return just the columns requested
            let batch = match self.projection.as_ref() {
                Some(columns) => batch.project(columns)?,
                None => batch.clone(),
            };

            Some(Ok(batch))
        } else {
            self.index = 1;
            let batch = &self.data[self.index - 1];

            // return just the columns requested
            let batch = match self.projection.as_ref() {
                Some(columns) => batch.project(columns)?,
                None => batch.clone(),
            };

            Some(Ok(batch))
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.data.len(), Some(self.data.len()))
    }
}

impl RecordBatchStream for EndlessMemoryStream {
    /// Get the schema
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }
}

/// Index into the data that has been returned so far
#[derive(Debug, Default, Clone)]
pub struct BatchIndex {
    inner: Arc<std::sync::Mutex<usize>>,
}

impl BatchIndex {
    /// Return the current index
    pub fn value(&self) -> usize {
        let inner = self.inner.lock().unwrap();
        *inner
    }

    // increment the current index by one
    pub fn incr(&self) {
        let mut inner = self.inner.lock().unwrap();
        *inner += 1;
    }
}

/// Iterator over batches
#[derive(Debug, Default)]
pub struct TestStream {
    /// Vector of record batches
    data: Vec<RecordBatch>,
    /// Index into the data that has been returned so far
    index: BatchIndex,
}

impl TestStream {
    /// Create an iterator for a vector of record batches. Assumes at
    /// least one entry in data (for the schema)
    pub fn new(data: Vec<RecordBatch>) -> Self {
        Self {
            data,
            ..Default::default()
        }
    }

    /// Return a handle to the index counter for this stream
    pub fn index(&self) -> BatchIndex {
        self.index.clone()
    }
}

impl Stream for TestStream {
    type Item = Result<RecordBatch>;

    fn poll_next(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let next_batch = self.index.value();

        Poll::Ready(if next_batch < self.data.len() {
            let next_batch = self.index.value();
            self.index.incr();
            Some(Ok(self.data[next_batch].clone()))
        } else {
            None
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.data.len(), Some(self.data.len()))
    }
}

impl RecordBatchStream for TestStream {
    /// Get the schema
    fn schema(&self) -> SchemaRef {
        self.data[0].schema()
    }
}

/// A Mock ExecutionPlan that can be used for writing tests of other
/// ExecutionPlans
#[derive(Debug)]
pub struct MockExec {
    /// the results to send back
    data: Vec<Result<RecordBatch>>,
    schema: SchemaRef,
    /// if true (the default), sends data using a separate task to ensure the
    /// batches are not available without this stream yielding first
    use_task: bool,
    cache: PlanProperties,
}

impl MockExec {
    /// Create a new `MockExec` with a single partition that returns
    /// the specified `Results`s.
    ///
    /// By default, the batches are not produced immediately (the
    /// caller has to actually yield and another task must run) to
    /// ensure any poll loops are correct. This behavior can be
    /// changed with `with_use_task`
    pub fn new(data: Vec<Result<RecordBatch>>, schema: SchemaRef) -> Self {
        let cache = Self::compute_properties(Arc::clone(&schema));
        Self {
            data,
            schema,
            use_task: true,
            cache,
        }
    }

    /// If `use_task` is true (the default) then the batches are sent
    /// back using a separate task to ensure the underlying stream is
    /// not immediately ready
    pub fn with_use_task(mut self, use_task: bool) -> Self {
        self.use_task = use_task;
        self
    }

    /// This function creates the cache object that stores the plan properties such as schema, equivalence properties, ordering, partitioning, etc.
    fn compute_properties(schema: SchemaRef) -> PlanProperties {
        let eq_properties = EquivalenceProperties::new(schema);

        PlanProperties::new(
            eq_properties,
            Partitioning::UnknownPartitioning(1),
            ExecutionMode::Bounded,
        )
    }
}

impl DisplayAs for MockExec {
    fn fmt_as(
        &self,
        t: DisplayFormatType,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(f, "MockExec")
            }
        }
    }
}

impl ExecutionPlan for MockExec {
    fn name(&self) -> &'static str {
        Self::static_name()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn properties(&self) -> &PlanProperties {
        &self.cache
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        _: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        unimplemented!()
    }

    /// Returns a stream which yields data
    fn execute(
        &self,
        partition: usize,
        _context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        assert_eq!(partition, 0);

        // Result doesn't implement clone, so do it ourself
        let data: Vec<_> = self
            .data
            .iter()
            .map(|r| match r {
                Ok(batch) => Ok(batch.clone()),
                Err(e) => Err(clone_error(e)),
            })
            .collect();

        if self.use_task {
            let mut builder = RecordBatchReceiverStream::builder(self.schema(), 2);
            // send data in order but in a separate task (to ensure
            // the batches are not available without the stream
            // yielding).
            let tx = builder.tx();
            builder.spawn(async move {
                for batch in data {
                    println!("Sending batch via delayed stream");
                    if let Err(e) = tx.send(batch).await {
                        println!("ERROR batch via delayed stream: {e}");
                    }
                }

                Ok(())
            });
            // returned stream simply reads off the rx stream
            Ok(builder.build())
        } else {
            // make an input that will error
            let stream = futures::stream::iter(data);
            Ok(Box::pin(RecordBatchStreamAdapter::new(
                self.schema(),
                stream,
            )))
        }
    }

    // Panics if one of the batches is an error
    fn statistics(&self) -> Result<Statistics> {
        let data: Result<Vec<_>> = self
            .data
            .iter()
            .map(|r| match r {
                Ok(batch) => Ok(batch.clone()),
                Err(e) => Err(clone_error(e)),
            })
            .collect();

        let data = data?;

        Ok(common::compute_record_batch_statistics(
            &[data],
            &self.schema,
            None,
        ))
    }
}

fn clone_error(e: &DataFusionError) -> DataFusionError {
    use DataFusionError::*;
    match e {
        Execution(msg) => Execution(msg.to_string()),
        _ => unimplemented!(),
    }
}

/// A Mock ExecutionPlan that does not start producing input until a
/// barrier is called
///
#[derive(Debug)]
pub struct BarrierExec {
    /// partitions to send back
    data: Vec<Vec<RecordBatch>>,
    schema: SchemaRef,

    /// all streams wait on this barrier to produce
    barrier: Arc<Barrier>,
    cache: PlanProperties,
}

impl BarrierExec {
    /// Create a new exec with some number of partitions.
    pub fn new(data: Vec<Vec<RecordBatch>>, schema: SchemaRef) -> Self {
        // wait for all streams and the input
        let barrier = Arc::new(Barrier::new(data.len() + 1));
        let cache = Self::compute_properties(Arc::clone(&schema), &data);
        Self {
            data,
            schema,
            barrier,
            cache,
        }
    }

    /// wait until all the input streams and this function is ready
    pub async fn wait(&self) {
        println!("BarrierExec::wait waiting on barrier");
        self.barrier.wait().await;
        println!("BarrierExec::wait done waiting");
    }

    /// This function creates the cache object that stores the plan properties such as schema, equivalence properties, ordering, partitioning, etc.
    fn compute_properties(
        schema: SchemaRef,
        data: &[Vec<RecordBatch>],
    ) -> PlanProperties {
        let eq_properties = EquivalenceProperties::new(schema);
        PlanProperties::new(
            eq_properties,
            Partitioning::UnknownPartitioning(data.len()),
            ExecutionMode::Bounded,
        )
    }
}

impl DisplayAs for BarrierExec {
    fn fmt_as(
        &self,
        t: DisplayFormatType,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(f, "BarrierExec")
            }
        }
    }
}

impl ExecutionPlan for BarrierExec {
    fn name(&self) -> &'static str {
        Self::static_name()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn properties(&self) -> &PlanProperties {
        &self.cache
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        unimplemented!()
    }

    fn with_new_children(
        self: Arc<Self>,
        _: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        unimplemented!()
    }

    /// Returns a stream which yields data
    fn execute(
        &self,
        partition: usize,
        _context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        assert!(partition < self.data.len());

        let mut builder = RecordBatchReceiverStream::builder(self.schema(), 2);

        // task simply sends data in order after barrier is reached
        let data = self.data[partition].clone();
        let b = Arc::clone(&self.barrier);
        let tx = builder.tx();
        builder.spawn(async move {
            println!("Partition {partition} waiting on barrier");
            b.wait().await;
            for batch in data {
                println!("Partition {partition} sending batch");
                if let Err(e) = tx.send(Ok(batch)).await {
                    println!("ERROR batch via barrier stream stream: {e}");
                }
            }

            Ok(())
        });

        // returned stream simply reads off the rx stream
        Ok(builder.build())
    }

    fn statistics(&self) -> Result<Statistics> {
        Ok(common::compute_record_batch_statistics(
            &self.data,
            &self.schema,
            None,
        ))
    }
}

/// A mock execution plan that errors on a call to execute
#[derive(Debug)]
pub struct ErrorExec {
    cache: PlanProperties,
}

impl Default for ErrorExec {
    fn default() -> Self {
        Self::new()
    }
}

impl ErrorExec {
    pub fn new() -> Self {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "dummy",
            DataType::Int64,
            true,
        )]));
        let cache = Self::compute_properties(schema);
        Self { cache }
    }

    /// This function creates the cache object that stores the plan properties such as schema, equivalence properties, ordering, partitioning, etc.
    fn compute_properties(schema: SchemaRef) -> PlanProperties {
        let eq_properties = EquivalenceProperties::new(schema);

        PlanProperties::new(
            eq_properties,
            Partitioning::UnknownPartitioning(1),
            ExecutionMode::Bounded,
        )
    }
}

impl DisplayAs for ErrorExec {
    fn fmt_as(
        &self,
        t: DisplayFormatType,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(f, "ErrorExec")
            }
        }
    }
}

impl ExecutionPlan for ErrorExec {
    fn name(&self) -> &'static str {
        Self::static_name()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn properties(&self) -> &PlanProperties {
        &self.cache
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        unimplemented!()
    }

    fn with_new_children(
        self: Arc<Self>,
        _: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        unimplemented!()
    }

    /// Returns a stream which yields data
    fn execute(
        &self,
        partition: usize,
        _context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        internal_err!("ErrorExec, unsurprisingly, errored in partition {partition}")
    }
}

/// A mock execution plan that simply returns the provided statistics
#[derive(Debug, Clone)]
pub struct StatisticsExec {
    stats: Statistics,
    schema: Arc<Schema>,
    cache: PlanProperties,
}
impl StatisticsExec {
    pub fn new(stats: Statistics, schema: Schema) -> Self {
        assert_eq!(
            stats
                .column_statistics.len(), schema.fields().len(),
            "if defined, the column statistics vector length should be the number of fields"
        );
        let cache = Self::compute_properties(Arc::new(schema.clone()));
        Self {
            stats,
            schema: Arc::new(schema),
            cache,
        }
    }

    /// This function creates the cache object that stores the plan properties such as schema, equivalence properties, ordering, partitioning, etc.
    fn compute_properties(schema: SchemaRef) -> PlanProperties {
        let eq_properties = EquivalenceProperties::new(schema);

        PlanProperties::new(
            eq_properties,
            Partitioning::UnknownPartitioning(2),
            ExecutionMode::Bounded,
        )
    }
}

impl DisplayAs for StatisticsExec {
    fn fmt_as(
        &self,
        t: DisplayFormatType,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(
                    f,
                    "StatisticsExec: col_count={}, row_count={:?}",
                    self.schema.fields().len(),
                    self.stats.num_rows,
                )
            }
        }
    }
}

impl ExecutionPlan for StatisticsExec {
    fn name(&self) -> &'static str {
        Self::static_name()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn properties(&self) -> &PlanProperties {
        &self.cache
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        _: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(self)
    }

    fn execute(
        &self,
        _partition: usize,
        _context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        unimplemented!("This plan only serves for testing statistics")
    }

    fn statistics(&self) -> Result<Statistics> {
        Ok(self.stats.clone())
    }
}

/// Execution plan that emits streams that block forever.
///
/// This is useful to test shutdown / cancelation behavior of certain execution plans.
#[derive(Debug)]
pub struct BlockingExec {
    /// Schema that is mocked by this plan.
    schema: SchemaRef,

    /// Ref-counting helper to check if the plan and the produced stream are still in memory.
    refs: Arc<()>,
    cache: PlanProperties,
}

impl BlockingExec {
    /// Create new [`BlockingExec`] with a give schema and number of partitions.
    pub fn new(schema: SchemaRef, n_partitions: usize) -> Self {
        let cache = Self::compute_properties(Arc::clone(&schema), n_partitions);
        Self {
            schema,
            refs: Default::default(),
            cache,
        }
    }

    /// Weak pointer that can be used for ref-counting this execution plan and its streams.
    ///
    /// Use [`Weak::strong_count`] to determine if the plan itself and its streams are dropped (should be 0 in that
    /// case). Note that tokio might take some time to cancel spawned tasks, so you need to wrap this check into a retry
    /// loop. Use [`assert_strong_count_converges_to_zero`] to archive this.
    pub fn refs(&self) -> Weak<()> {
        Arc::downgrade(&self.refs)
    }

    /// This function creates the cache object that stores the plan properties such as schema, equivalence properties, ordering, partitioning, etc.
    fn compute_properties(schema: SchemaRef, n_partitions: usize) -> PlanProperties {
        let eq_properties = EquivalenceProperties::new(schema);

        PlanProperties::new(
            eq_properties,
            Partitioning::UnknownPartitioning(n_partitions),
            ExecutionMode::Bounded,
        )
    }
}

impl DisplayAs for BlockingExec {
    fn fmt_as(
        &self,
        t: DisplayFormatType,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(f, "BlockingExec",)
            }
        }
    }
}

impl ExecutionPlan for BlockingExec {
    fn name(&self) -> &'static str {
        Self::static_name()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn properties(&self) -> &PlanProperties {
        &self.cache
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        // this is a leaf node and has no children
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        _: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        internal_err!("Children cannot be replaced in {self:?}")
    }

    fn execute(
        &self,
        _partition: usize,
        _context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        Ok(Box::pin(BlockingStream {
            schema: Arc::clone(&self.schema),
            _refs: Arc::clone(&self.refs),
        }))
    }
}

/// A [`RecordBatchStream`] that is pending forever.
#[derive(Debug)]
pub struct BlockingStream {
    /// Schema mocked by this stream.
    schema: SchemaRef,

    /// Ref-counting helper to check if the stream are still in memory.
    _refs: Arc<()>,
}

impl Stream for BlockingStream {
    type Item = Result<RecordBatch>;

    fn poll_next(
        self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        Poll::Pending
    }
}

impl RecordBatchStream for BlockingStream {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }
}

/// Asserts that the strong count of the given [`Weak`] pointer converges to zero.
///
/// This might take a while but has a timeout.
pub async fn assert_strong_count_converges_to_zero<T>(refs: Weak<T>) {
    tokio::time::timeout(std::time::Duration::from_secs(10), async {
        loop {
            if dbg!(Weak::strong_count(&refs)) == 0 {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
}

///

/// Execution plan that emits streams that panics.
///
/// This is useful to test panic handling of certain execution plans.
#[derive(Debug)]
pub struct PanicExec {
    /// Schema that is mocked by this plan.
    schema: SchemaRef,

    /// Number of output partitions. Each partition will produce this
    /// many empty output record batches prior to panicking
    batches_until_panics: Vec<usize>,
    cache: PlanProperties,
}

impl PanicExec {
    /// Create new [`PanicExec`] with a give schema and number of
    /// partitions, which will each panic immediately.
    pub fn new(schema: SchemaRef, n_partitions: usize) -> Self {
        let batches_until_panics = vec![0; n_partitions];
        let cache = Self::compute_properties(Arc::clone(&schema), &batches_until_panics);
        Self {
            schema,
            batches_until_panics,
            cache,
        }
    }

    /// Set the number of batches prior to panic for a partition
    pub fn with_partition_panic(mut self, partition: usize, count: usize) -> Self {
        self.batches_until_panics[partition] = count;
        self
    }

    /// This function creates the cache object that stores the plan properties such as schema, equivalence properties, ordering, partitioning, etc.
    fn compute_properties(
        schema: SchemaRef,
        batches_until_panics: &[usize],
    ) -> PlanProperties {
        let eq_properties = EquivalenceProperties::new(schema);
        let num_partitions = batches_until_panics.len();

        PlanProperties::new(
            eq_properties,
            Partitioning::UnknownPartitioning(num_partitions),
            ExecutionMode::Bounded,
        )
    }
}

impl DisplayAs for PanicExec {
    fn fmt_as(
        &self,
        t: DisplayFormatType,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(f, "PanicExec",)
            }
        }
    }
}

impl ExecutionPlan for PanicExec {
    fn name(&self) -> &'static str {
        Self::static_name()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn properties(&self) -> &PlanProperties {
        &self.cache
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        // this is a leaf node and has no children
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        _: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        internal_err!("Children cannot be replaced in {:?}", self)
    }

    fn execute(
        &self,
        partition: usize,
        _context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        Ok(Box::pin(PanicStream {
            partition,
            batches_until_panic: self.batches_until_panics[partition],
            schema: Arc::clone(&self.schema),
            ready: false,
        }))
    }
}

/// A [`RecordBatchStream`] that yields every other batch and panics
/// after `batches_until_panic` batches have been produced.
///
/// Useful for testing the behavior of streams on panic
#[derive(Debug)]
struct PanicStream {
    /// Which partition was this
    partition: usize,
    /// How may batches will be produced until panic
    batches_until_panic: usize,
    /// Schema mocked by this stream.
    schema: SchemaRef,
    /// Should we return ready ?
    ready: bool,
}

impl Stream for PanicStream {
    type Item = Result<RecordBatch>;

    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        if self.batches_until_panic > 0 {
            if self.ready {
                self.batches_until_panic -= 1;
                self.ready = false;
                let batch = RecordBatch::new_empty(Arc::clone(&self.schema));
                return Poll::Ready(Some(Ok(batch)));
            } else {
                self.ready = true;
                // get called again
                cx.waker().wake_by_ref();
                return Poll::Pending;
            }
        }
        panic!("PanickingStream did panic: {}", self.partition)
    }
}

impl RecordBatchStream for PanicStream {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }
}
