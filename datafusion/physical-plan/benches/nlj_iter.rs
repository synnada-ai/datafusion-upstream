use arrow_array::{Int32Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema, SchemaRef, SortOptions};
use criterion::async_executor::FuturesExecutor;
use criterion::{
    black_box, criterion_group, criterion_main, Bencher, BenchmarkId, Criterion,
    PlotConfiguration,
};
use datafusion_common::{JoinSide, JoinType, ScalarValue};
use datafusion_execution::TaskContext;
use datafusion_expr::Operator;
use datafusion_physical_expr::expressions::{col, BinaryExpr, Literal};
use datafusion_physical_expr_common::sort_expr::{LexOrdering, PhysicalSortExpr};
use datafusion_physical_plan::joins::utils::{ColumnIndex, JoinFilter};
use datafusion_physical_plan::joins::NestedLoopJoinExec;
use datafusion_physical_plan::memory::MemoryExec;
use datafusion_physical_plan::{execute_stream, ExecutionPlan};
use futures::TryStreamExt;
use itertools::{iproduct, Itertools};
use std::fs::File;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Hash)]
struct TestParams {
    left_column_count: usize,
    right_column_count: usize,
    left_rows: usize,
    right_rows: usize,
    batch_size: usize,
    selectivity: i32,
    join_type: JoinType,
    right_ordered: bool,
    use_iterator_impl: bool,
}

impl TestParams {
    fn left_column_count(mut self, left_column_count: usize) -> Self {
        self.left_column_count = left_column_count;
        self
    }

    fn right_column_count(mut self, right_column_count: usize) -> Self {
        self.right_column_count = right_column_count;
        self
    }

    fn left_rows(mut self, left_rows: usize) -> Self {
        self.left_rows = left_rows;
        self
    }

    fn right_rows(mut self, right_rows: usize) -> Self {
        self.right_rows = right_rows;
        self
    }

    fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    fn selectivity(mut self, selectivity: i32) -> Self {
        self.selectivity = selectivity;
        self
    }

    fn join_type(mut self, join_type: JoinType) -> Self {
        self.join_type = join_type;
        self
    }

    fn right_ordered(mut self, right_ordered: bool) -> Self {
        self.right_ordered = right_ordered;
        self
    }

    fn use_iterator_impl(mut self, use_iterator_impl: bool) -> Self {
        self.use_iterator_impl = use_iterator_impl;
        self
    }
}

impl Default for TestParams {
    fn default() -> Self {
        Self {
            left_column_count: 1,
            right_column_count: 1,
            left_rows: 1024,
            right_rows: 8192,
            batch_size: 128,
            selectivity: 20,
            join_type: JoinType::Inner,
            right_ordered: false,
            use_iterator_impl: false,
        }
    }
}

fn criterion_benchmark(c: &mut Criterion) {
    for column_count in [8] {
        for selectivity in [0, 5, 10, 15, 20, 25, 30] {
            let mut g = c.benchmark_group(format!(
                "column_count_{}_selectivity_{}",
                column_count, selectivity
            ));
            let plot_cfg = PlotConfiguration::default()
                .summary_scale(criterion::AxisScale::Logarithmic);
            g.plot_config(plot_cfg);
            for i in (0..=13) {
                let batch_size = 1 << i;
                g.bench_with_input(
                    BenchmarkId::new("indices", batch_size),
                    &batch_size,
                    |b, &batch_size| {
                        let test_params = TestParams::default()
                            .batch_size(batch_size)
                            .left_column_count(column_count / 2)
                            .right_column_count(column_count / 2)
                            .selectivity(selectivity)
                            .use_iterator_impl(false);
                        bench(b, test_params);
                    },
                );
            }
        }
    }
}

fn bench(b: &mut Bencher, test_params: TestParams) {
    let nlj = prepare_plan(test_params);
    b.to_async(FuturesExecutor).iter(|| async {
        let ctx = Arc::new(TaskContext::default());
        let stream = execute_stream(nlj.clone(), ctx).unwrap();
        let a: Vec<_> = stream.try_collect().await.unwrap();
        black_box(a);
    })
}

fn prepare_plan(test_params: TestParams) -> Arc<dyn ExecutionPlan> {
    // a int, b int ordered
    let left_schema = create_schema("a", test_params.left_column_count);
    let right_schema = create_schema("b", test_params.right_column_count);
    let merged_schema = Schema::new(
        left_schema
            .fields()
            .iter()
            .chain(right_schema.fields())
            .cloned()
            .collect::<Vec<_>>(),
    );

    let left = Arc::new(
        MemoryExec::try_new(
            &[gen_data(
                left_schema.clone(),
                test_params.left_rows,
                test_params.batch_size,
            )],
            left_schema.clone(),
            None,
        )
        .unwrap(),
    );
    let right = if test_params.right_ordered {
        Arc::new(
            MemoryExec::try_new(
                &[gen_data(
                    right_schema.clone(),
                    test_params.right_rows,
                    test_params.batch_size,
                )],
                right_schema.clone(),
                None,
            )
            .unwrap()
            .try_with_sort_information(vec![LexOrdering::from(vec![
                PhysicalSortExpr::new(
                    col("b1", &right_schema).unwrap(),
                    SortOptions::default(),
                ),
            ])])
            .unwrap(),
        )
    } else {
        Arc::new(
            MemoryExec::try_new(
                &[gen_data(
                    right_schema.clone(),
                    test_params.right_rows,
                    test_params.batch_size,
                )],
                right_schema.clone(),
                None,
            )
            .unwrap(),
        )
    };

    // a > b

    // add all columns
    let mut sum = merged_schema.fields().iter().fold(
        col("a1", &left_schema).unwrap(),
        |acc, field| {
            Arc::new(BinaryExpr::new(
                acc,
                Operator::Plus,
                col(field.name(), &merged_schema).unwrap(),
            ))
        },
    );

    let modulo = Arc::new(BinaryExpr::new(
        sum,
        Operator::Modulo,
        Arc::new(Literal::new(ScalarValue::Int32(Some(100)))),
    ));
    let comparison = Arc::new(BinaryExpr::new(
        modulo,
        Operator::LtEq,
        Arc::new(Literal::new(ScalarValue::Int32(Some(
            test_params.selectivity,
        )))),
    ));
    // let comparison = Arc::new(BinaryExpr::new(
    //     col("a1", &left_schema).unwrap(),
    //     Operator::Gt,
    //     col("b1", &right_schema).unwrap(),
    // ));

    let filter = Some(JoinFilter::new(
        comparison,
        (0..test_params.left_column_count)
            .map(|index| ColumnIndex {
                index,
                side: JoinSide::Left,
            })
            .chain(
                (0..test_params.right_column_count).map(|index| ColumnIndex {
                    index,
                    side: JoinSide::Right,
                }),
            )
            .collect(),
        merged_schema,
    ));
    Arc::new(
        NestedLoopJoinExec::try_new(left, right, filter, &test_params.join_type).unwrap(),
    )
}

fn create_schema(prefix: &str, column_count: usize) -> SchemaRef {
    Arc::new(Schema::new(
        (1..=column_count)
            .map(|i| Field::new(&format!("{}{}", prefix, i), DataType::Int32, false))
            .collect::<Vec<_>>(),
    ))
}

fn gen_data(schema: SchemaRef, row_count: usize, batch_size: usize) -> Vec<RecordBatch> {
    // 0..n-1 in BATCH_SIZE chunks
    (0..row_count as i32)
        .chunks(batch_size)
        .into_iter()
        .map(|chunk| {
            let chunk = chunk.collect::<Vec<_>>();
            RecordBatch::try_new(
                schema.clone(),
                (0..schema.fields().len())
                    .map(|i| {
                        Arc::new(Int32Array::from(chunk.clone()))
                            as Arc<dyn arrow_array::Array>
                    })
                    .collect::<Vec<_>>(),
            )
            .unwrap()
        })
        .collect()
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
