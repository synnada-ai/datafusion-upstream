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

#[macro_use]
extern crate criterion;
use criterion::Criterion;
extern crate arrow;
extern crate datafusion;

use std::sync::Arc;

use arrow::array::Int64Array;
use arrow::{array::StringArray, record_batch::RecordBatch};
use datafusion::physical_plan::{
    collect,
    expressions::{col, PhysicalSortExpr},
};
use datafusion::prelude::SessionContext;
use datafusion_datasource::memory::MemorySourceConfig;
use datafusion_datasource::source::DataSourceExec;
use datafusion_execution::TaskContext;
use datafusion_physical_expr::Partitioning;
use datafusion_physical_expr_common::sort_expr::LexOrdering;
use datafusion_physical_plan::repartition::RepartitionExec;
use datafusion_physical_plan::sorts::sort::SortExec;
use datafusion_physical_plan::stream::RecordBatchReceiverStream;
use datafusion_physical_plan::ExecutionPlan;

use futures::StreamExt;
use rand::Rng;
use tokio::runtime::Runtime;

const BATCH_SIZE: usize = 1000;

fn repartition_with_preserve_order_plan(
    session_ctx: Arc<SessionContext>,
    data: &[Vec<RecordBatch>],
    n_input_partition: usize,
    n_output_partition: usize,
) {
    let schema = data[0][0].schema();

    let sort = LexOrdering::new(vec![PhysicalSortExpr {
        expr: col("a", &schema).unwrap(),
        options: Default::default(),
    }]);

    let mut source = MemorySourceConfig::try_new(data, schema, None).unwrap();
    source = source
        .try_with_sort_information(vec![sort.clone()])
        .unwrap();
    let exec = Arc::new(
        DataSourceExec::new(Arc::new(source))
            .with_partitioning(Partitioning::UnknownPartitioning(n_input_partition)),
    );
    let plan = Arc::new(
        RepartitionExec::try_new(exec, Partitioning::RoundRobinBatch(n_output_partition))
            .unwrap()
            .with_preserve_order(),
    );
    let task_ctx = session_ctx.task_ctx();
    let rt = Runtime::new().unwrap();
    if n_output_partition == 1 {
        rt.block_on(collect(plan, task_ctx)).unwrap();
    } else {
        rt.block_on(collect_async(plan, n_output_partition, task_ctx))
    }
}

async fn collect_async(
    plan: Arc<dyn ExecutionPlan>,
    n_output_partition: usize,
    task_ctx: Arc<TaskContext>,
) {
    let mut builder =
        RecordBatchReceiverStream::builder(plan.schema(), n_output_partition);
    for part_i in 0..n_output_partition {
        let context = Arc::clone(&task_ctx);
        let plan_captured = Arc::clone(&plan);
        builder.spawn(async move {
            let mut stream = match plan_captured.execute(part_i, Arc::clone(&context)) {
                Err(e) => {
                    unreachable!("Error happened during stream initialization {}", e);
                }
                Ok(stream) => stream,
            };

            while stream.next().await.is_some() {
                continue;
            }

            Ok(())
        });
    }
    let mut stream = builder.build();
    while stream.next().await.is_some() {
        continue;
    }
}

fn repartition_with_sort_exec_plan(
    session_ctx: Arc<SessionContext>,
    data: &[Vec<RecordBatch>],
    n_input_partition: usize,
    n_output_partition: usize,
) {
    let schema = data[0][0].schema();
    let sort = LexOrdering::new(vec![PhysicalSortExpr {
        expr: col("a", &schema).unwrap(),
        options: Default::default(),
    }]);

    let mut source = MemorySourceConfig::try_new(data, schema, None).unwrap();
    source = source
        .try_with_sort_information(vec![sort.clone()])
        .unwrap();
    let exec = Arc::new(
        DataSourceExec::new(Arc::new(source))
            .with_partitioning(Partitioning::UnknownPartitioning(n_input_partition)),
    );
    let repartition = Arc::new(
        RepartitionExec::try_new(exec, Partitioning::RoundRobinBatch(n_output_partition))
            .unwrap(),
    );
    let plan = Arc::new(SortExec::new(sort.clone(), repartition));
    let task_ctx = session_ctx.task_ctx();
    let rt = Runtime::new().unwrap();
    if n_output_partition == 1 {
        rt.block_on(collect(plan, task_ctx)).unwrap();
    } else {
        rt.block_on(collect_async(plan, n_output_partition, task_ctx))
    }
}

fn str_batches(probability: f64, record_batch_count: usize) -> Vec<RecordBatch> {
    let mut rbs = Vec::with_capacity(record_batch_count);
    let mut rng = rand::thread_rng();

    let mut latest_index = 0;

    for _ in 0..record_batch_count {
        let mut col_a = Vec::with_capacity(BATCH_SIZE);
        for _ in 0..BATCH_SIZE {
            latest_index = if rng.gen_bool(probability) {
                latest_index
            } else {
                let random_increment = rng.gen_range(1..=5);
                latest_index + random_increment
            };
            col_a.push(Some(format!("a-{latest_index:?}")));
        }
        let rb = RecordBatch::try_from_iter(vec![(
            "a",
            Arc::new(StringArray::from_iter(col_a)) as _,
        )])
        .unwrap();
        rbs.push(rb);
    }
    rbs
}

fn int_batches(probability: f64, record_batch_count: usize) -> Vec<RecordBatch> {
    let mut rbs = Vec::with_capacity(record_batch_count);
    let mut rng = rand::thread_rng();

    let mut latest_index: i64 = 0;

    for _ in 0..record_batch_count {
        let mut col_a = Vec::with_capacity(BATCH_SIZE);
        for _ in 0..BATCH_SIZE {
            latest_index = if rng.gen_bool(probability) {
                latest_index
            } else {
                let random_increment = rng.gen_range(1..=5);
                latest_index + random_increment
            };
            col_a.push(Some(latest_index));
        }
        let array = Int64Array::from(col_a);
        let rb = RecordBatch::try_from_iter(vec![("a", Arc::new(array) as _)]).unwrap();
        rbs.push(rb);
    }
    rbs
}

fn split_batches_into_parts(
    batches: Vec<RecordBatch>,
    n_input_partition: usize,
) -> Vec<Vec<RecordBatch>> {
    if n_input_partition == 1 {
        return batches.into_iter().map(|rb| vec![rb]).collect::<Vec<_>>();
    }

    let mut data: Vec<Vec<RecordBatch>> = Vec::with_capacity(n_input_partition);
    for (i, batch) in batches.iter().enumerate() {
        let partition = i % n_input_partition;
        if data.len() <= partition {
            data.push(vec![]);
        }
        data[partition].push(batch.clone());
    }
    data.clone()
}

fn criterion_benchmark(c: &mut Criterion) {
    let input_output_partitions = vec![(8, 8), (8, 32), (32, 32), (32, 8)];
    let probabilities = vec![0.0, 0.1, 0.5, 0.99];
    let record_batch_counts = vec![5000, 20000];
    let is_ints = vec![true, false];

    let mut benches = vec![];
    for input_output_part in input_output_partitions {
        for probability in probabilities.clone() {
            for rbc in record_batch_counts.clone() {
                for is_int in is_ints.clone() {
                    let bench = if is_int {
                        (
                            format!(
                                "%{}_prob_{}_to_{}_n_{}_int",
                                probability * 100.0,
                                input_output_part.0,
                                input_output_part.1,
                                rbc
                            ),
                            int_batches(probability, rbc),
                            input_output_part.0,
                            input_output_part.1,
                        )
                    } else {
                        (
                            format!(
                                "%{}_prob_{}_to_{}_n_{}_str",
                                probability * 100.0,
                                input_output_part.0,
                                input_output_part.1,
                                rbc
                            ),
                            str_batches(probability, rbc),
                            input_output_part.0,
                            input_output_part.1,
                        )
                    };
                    benches.push(bench)
                }
            }
        }
    }

    // Benches:
    // ("%10_prob_8_to_8_n_100_str", str_batches(0.1, 100), 8, 8),
    // ("%10_prob_8_to_32_n_100_int", int_batches(0.1, 100), 8, 32),

    let ctx = Arc::new(SessionContext::new());
    for (name, input, n_input_partition, n_output_partition) in benches {
        let ctx_clone = ctx.clone();
        let data = split_batches_into_parts(input, n_input_partition);
        c.bench_function(format!("{name} with preserve order").as_str(), |b| {
            b.iter(|| {
                repartition_with_preserve_order_plan(
                    ctx_clone.clone(),
                    &data,
                    n_input_partition,
                    n_output_partition,
                )
            });
        });

        c.bench_function(format!("{name} with sort exec").as_str(), |b| {
            b.iter(|| {
                repartition_with_sort_exec_plan(
                    ctx_clone.clone(),
                    &data,
                    n_input_partition,
                    n_output_partition,
                )
            });
        });
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
