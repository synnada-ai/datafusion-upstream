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
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS
// OF ANY KIND, either express or implied.  See the License
// for the specific language governing permissions and
// limitations under the License.

#[macro_use]
extern crate criterion;
extern crate arrow;
extern crate datafusion;

use arrow::array::{ArrayRef, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use criterion::{BenchmarkGroup, Criterion};
use datafusion::error::Result;
use datafusion::execution::context::SessionContext;
use parking_lot::Mutex;
use rand::Rng;
use std::sync::Arc;
use tokio::runtime::Runtime;

fn generate_test_data(num_rows: usize) -> RecordBatch {
    let mut rng = rand::thread_rng();
    let data: Vec<u64> = (0..num_rows)
        .map(|_| rng.gen_range(0..100_000_000))
        .collect();
    let array: ArrayRef = Arc::new(UInt64Array::from(data));
    let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::UInt64, false)]));
    RecordBatch::try_new(schema, vec![array]).unwrap()
}

fn create_context_with_table() -> Result<Arc<Mutex<SessionContext>>> {
    let ctx = SessionContext::new();
    let batch = generate_test_data(100_000);
    ctx.register_batch("random_a", batch)?;
    Ok(Arc::new(Mutex::new(ctx)))
}

fn benchmark_query(ctx: Arc<Mutex<SessionContext>>) {
    let query = "SELECT a > -1 FROM random_a";
    let rt = Runtime::new().unwrap();
    let df = rt.block_on(ctx.lock().sql(query)).unwrap();
    criterion::black_box(rt.block_on(df.collect()).unwrap());
}

fn criterion_benchmark(c: &mut Criterion) {
    let ctx = create_context_with_table().unwrap();

    let mut group: BenchmarkGroup<_> = c.benchmark_group("");
    group.bench_function("SELECT a > -1", |b| {
        b.iter(|| {
            benchmark_query(ctx.clone());
        });
    });

    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
