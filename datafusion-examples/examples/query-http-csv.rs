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

use datafusion::error::Result;
use datafusion::prelude::*;
use object_store::http::HttpBuilder;
use std::sync::Arc;
use url::Url;

/// This example demonstrates executing a simple query against an Arrow data source (CSV) and
/// fetching results
#[tokio::main]
async fn main() -> Result<()> {
    // create local execution context
    let ctx = SessionContext::new();

    // setup http object store
    let base_url = Url::parse("https://github.com").unwrap();
    let http_store = HttpBuilder::new()
        .with_url(base_url.clone())
        .build()
        .unwrap();
    ctx.register_object_store(&base_url, Arc::new(http_store));

    // register csv file with the execution context
    ctx.register_csv(
        "aggregate_test_100",
        "/Users/kamille/Desktop/github/datafusion/testing/data/csv/aggregate_test_100.csv",
        CsvReadOptions::new(),
    )
    .await?;

    // execute the query
    // let df = ctx
    //     .sql("SELECT c1, c2, MIN(c3) + SUM(c3) + 4 FROM aggregate_test_100 GROUP BY c1, c2 LIMIT 1")
    //     .await?;
    let df = ctx
        .sql("SELECT c1, c2, MIN(c3) + 3 FROM aggregate_test_100 GROUP BY c1, c2 LIMIT 1")
        .await?;

    let plan = df.logical_plan();
    println!("{plan}");
    // print the results
    df.show().await?;

    // let df = ctx
    // .sql("SELECT c1, c2, MIN(c3) FROM aggregate_test_100 GROUP BY GROUPING SETS ((c1,c2),(c1),(c2)) LIMIT 1")
    // .await?;

    // // print the results
    // df.show().await?;

    Ok(())
}
