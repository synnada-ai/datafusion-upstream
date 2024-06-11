// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use itertools::izip;
use std::sync::Arc;

use crate::error::Result;
use crate::physical_optimizer::PhysicalOptimizerRule;
use crate::physical_plan::ExecutionPlan;

use datafusion_common::config::ConfigOptions;
use datafusion_common::plan_err;
use datafusion_common::tree_node::{Transformed, TransformedResult, TreeNode};
use datafusion_physical_plan::ExecutionPlanProperties;

#[derive(Default)]
pub struct SanityCheckPlan {}

impl SanityCheckPlan {
    #[allow(missing_docs)]
    pub fn new() -> Self {
        Self {}
    }
}

impl PhysicalOptimizerRule for SanityCheckPlan {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        plan.transform_down(|p| check_plan_sanity(p)).data()
    }

    fn name(&self) -> &str {
        "SanityCheckPlan"
    }

    fn schema_check(&self) -> bool {
        true
    }
}

/// Ensures that the plan is pipeline friendly and the order and
/// distribution requirements from its children are satisfied.
pub fn check_plan_sanity(
    plan: Arc<dyn ExecutionPlan>,
) -> Result<Transformed<Arc<dyn ExecutionPlan>>> {
    if !plan.execution_mode().pipeline_friendly() {
        return plan_err!("Plan {:?} is not pipeline friendly.", plan)
    }

    for (child, child_sort_req, child_dist_req) in izip!(
        plan.children().iter(),
        plan.required_input_ordering().iter(),
        plan.required_input_distribution().iter()
    ) {
        let child_eq_props = child.equivalence_properties();
        match child_sort_req {
            None => (),
            Some(child_sort_req) => {
                if !child_eq_props.ordering_satisfy_requirement(&child_sort_req) {
                    return plan_err!(
                        "Child: {:?} does not satisfy parent order requirements",
                        child
                    );
                }
            }
        };

        if !child
            .output_partitioning()
            .satisfy(&child_dist_req, child_eq_props)
        {
            return plan_err!(
                "Child: {:?} does not satisfy parent distribution requirements",
                child
            );
        }
    }

    Ok(Transformed::no(plan))
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::physical_optimizer::test_utils::{
        bounded_window_exec, memory_exec, sort_exec, sort_expr_options,
    };
    use arrow::compute::SortOptions;
    use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
    use datafusion_common::Result;

    fn create_test_schema() -> Result<SchemaRef> {
        let c9_column = Field::new("c9", DataType::Int32, true);
        let schema = Arc::new(Schema::new(vec![c9_column]));
        Ok(schema)
    }

    #[tokio::test]
    async fn test_bw_sort_requirement() -> Result<()> {
        let schema = create_test_schema()?;
        let source = memory_exec(&schema);
        let sort_exprs = vec![sort_expr_options(
            "c9",
            &source.schema(),
            SortOptions {
                descending: false,
                nulls_first: false,
            },
        )];
        let sort = sort_exec(sort_exprs.clone(), source);
        let bw = bounded_window_exec("c9", sort_exprs, sort);
        let sanity_checker = SanityCheckPlan::new();
        let opts = ConfigOptions::default();
        assert_eq!(sanity_checker.optimize(bw, &opts).is_ok(), true);
        Ok(())
    }

    #[tokio::test]
    async fn test_bw_no_sort_requirement() -> Result<()> {
        let schema = create_test_schema()?;
        let source = memory_exec(&schema);
        let sort_exprs = vec![sort_expr_options(
            "c9",
            &source.schema(),
            SortOptions {
                descending: false,
                nulls_first: false,
            },
        )];
        let bw = bounded_window_exec("c9", sort_exprs, source);
        let sanity_checker = SanityCheckPlan::new();
        let opts = ConfigOptions::default();
        assert_eq!(sanity_checker.optimize(bw, &opts).is_ok(), false);
        Ok(())
    }
}
