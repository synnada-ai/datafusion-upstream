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

//! [`EliminateGroupByConstant`] removes constant expressions from `GROUP BY` clause
use std::sync::Arc;

use crate::optimizer::ApplyOrder;
use crate::{OptimizerConfig, OptimizerRule};

use datafusion_common::tree_node::{
    Transformed, TreeNode, TreeNodeRecursion, TreeNodeRewriter,
};
use datafusion_common::{internal_err, DFSchema, HashSet, Result};
use datafusion_expr::expr::{AggregateFunction, AggregateFunctionParams, Alias};
use datafusion_expr::{
    col, Aggregate, BinaryExpr, Expr, ExprSchemable, LogicalPlan, LogicalPlanBuilder,
    Operator, Volatility,
};

/// Optimizer rule that removes constant expressions from `GROUP BY` clause
/// and places additional projection on top of aggregation, to preserve
/// original schema
#[derive(Default, Debug)]
pub struct TransformLinearAggregation {}

impl TransformLinearAggregation {
    pub fn new() -> Self {
        Self {}
    }
}

impl OptimizerRule for TransformLinearAggregation {
    fn supports_rewrite(&self) -> bool {
        true
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        _config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>> {
        match plan {
            LogicalPlan::Aggregate(aggregate) => {
                // TODO: transform linear udaf in group by cases
                // Currently, we only try to transform `aggr_expr` in no grouping cases,
                // and surely `aggr_expr` should exist
                if !aggregate.group_expr.is_empty() || aggregate.aggr_expr.is_empty() {
                    return Ok(Transformed::no(LogicalPlan::Aggregate(aggregate)));
                }

                // Try to transform `aggr_expr`
                let mut new_aggr_exprs =
                    HashSet::with_capacity(aggregate.aggr_expr.len());
                let mut new_proj_exprs = Vec::new();
                let mut transformed = false;
                let input_schema = aggregate.input.schema();
                for expr in aggregate.aggr_expr.iter() {
                    transformed |= maybe_transform_aggr_expr(
                        expr,
                        input_schema,
                        &mut new_aggr_exprs,
                        &mut new_proj_exprs,
                    )?;
                }

                // If transform happened, we need to rewrite the old `Aggregate`, otherwise we just
                // return the old one
                if !transformed {
                    return Ok(Transformed::no(LogicalPlan::Aggregate(aggregate)));
                }

                let transformed_aggregate = LogicalPlan::Aggregate(Aggregate::try_new(
                    aggregate.input,
                    aggregate.group_expr.clone(),
                    new_aggr_exprs.into_iter().collect(),
                )?);

                let projection_expr =
                    aggregate.group_expr.into_iter().chain(new_proj_exprs);

                let projection = LogicalPlanBuilder::from(transformed_aggregate)
                    .project(projection_expr)?
                    .build()?;

                Ok(Transformed::yes(projection))
            }
            _ => Ok(Transformed::no(plan)),
        }
    }

    fn name(&self) -> &str {
        "transform_linear_aggregation"
    }

    fn apply_order(&self) -> Option<ApplyOrder> {
        Some(ApplyOrder::BottomUp)
    }
}

/// Try to transform the udaf
///
/// It will try to transform if `aggr_expr` satisfies following requirements:
///   - It is a `linear` udaf
///   - It should be `not distinct`, `not filter exists`, `not order by exists`
///   - Only one arg exist in `args`, and it is `non-nullable`
///
/// For transformation, we will found the `plus` and `multiply` binary expr,
/// and then transform.
///
/// Examples:
///
/// ```text
///   aggr(a + b + c) --> aggr(a) + aggr(b) + aggr(c)
///   aggr(3 * a) --> 3 * aggr(a)
///   aggr(3 * (a + b)) --> 3 * (aggr(a) + aggr(b))
/// ```
///
fn maybe_transform_aggr_expr(
    aggr_expr: &Expr,
    input_schema: &DFSchema,
    new_aggr_exprs: &mut HashSet<Expr>,
    new_proj_exprs: &mut Vec<Expr>,
) -> Result<bool> {
    let Expr::AggregateFunction(aggr_function) = aggr_expr else {
        return internal_err!(
            "expect Expr::AggregateFunction in aggr_expr, but found:{aggr_expr:?}"
        );
    };

    let AggregateFunctionParams {
        args,
        distinct,
        filter,
        order_by,
        ..
    } = &aggr_function.params;

    // Check if we should try to transform
    if !aggr_function.func.is_linear()
        || *distinct
        || filter.is_some()
        || order_by.is_some()
        || args.len() != 1
        || args[0].nullable(input_schema)?
    {
        new_aggr_exprs.insert(aggr_expr.clone());
        new_proj_exprs.push(aggr_expr.clone());
        return Ok(false);
    }

    // Try rewriting to get the `new_aggr_expr`s (they will be combined to be `maybe_transformed_expr`)
    let mut rewriter = LinearAggregationRewriter::new(new_aggr_exprs, aggr_function);
    let maybe_transformed_expr = args[0].clone().rewrite(&mut rewriter)?.data;

    // Check if `new_aggr_expr`s generated, if so the combined expr `maybe_transformed_expr`
    // should not equal to `aggr_expr`
    let transformed = &maybe_transformed_expr != aggr_expr;

    // Make projection according to if it is transformed:
    //   - If so, we alias `maybe_transformed_expr` to old name to let parent can still refer
    //   - If not, we just keep the old projection
    if transformed {
        // Generate the new project to let parent can still refer
        let (_, proj_name) = aggr_expr.qualified_name();
        let new_proj_expr = maybe_transformed_expr.alias(proj_name);
        new_proj_exprs.push(new_proj_expr);
    } else {
        new_proj_exprs.push(aggr_expr.clone());
    }

    Ok(transformed)
}

struct LinearAggregationRewriter<'a> {
    new_aggr_exprs: &'a mut HashSet<Expr>,
    origin_aggr_function: &'a AggregateFunction,
}

impl<'a> LinearAggregationRewriter<'a> {
    pub fn new(
        new_aggr_exprs: &'a mut HashSet<Expr>,
        origin_aggr_function: &'a AggregateFunction,
    ) -> Self {
        Self {
            new_aggr_exprs,
            origin_aggr_function,
        }
    }
}

impl<'a> TreeNodeRewriter for LinearAggregationRewriter<'a> {
    type Node = Expr;

    fn f_down(&mut self, node: Self::Node) -> Result<Transformed<Self::Node>> {
        let transformed = match &node {
            // In these two situations, the node should be the parent and we recursively
            // traverse their children to perform transformation:
            //   - Plus case, aggr(a + b) => aggr(a) + aggr(b)
            //   - Multiply case, aggr(literal * a) => literal * aggr(a)

            // Plus expr case
            // We traverse and create `new aggr expr` for its `left` and `right`
            Expr::BinaryExpr(BinaryExpr {
                op: Operator::Plus, ..
            }) => Transformed::no(node),

            // Multiply expr case
            // We can only transform the case that one side in node is `literal`.
            // And we traverse and create `new aggr expr` for `non-literal` side
            Expr::BinaryExpr(BinaryExpr { left, op, right })
                if matches!(op, Operator::Multiply)
                    && !matches!(left.as_ref(), &Expr::Literal(_))
                    && matches!(right.as_ref(), &Expr::Literal(_)) =>
            {
                let new_left = left.clone().rewrite(self)?.data;
                let new_expr = BinaryExpr::new(Box::new(new_left), *op, right.clone());
                Transformed::new(
                    Expr::BinaryExpr(new_expr),
                    true,
                    TreeNodeRecursion::Jump,
                )
            }

            Expr::BinaryExpr(BinaryExpr { left, op, right })
                if matches!(op, Operator::Multiply)
                    && matches!(left.as_ref(), &Expr::Literal(_))
                    && !matches!(right.as_ref(), &Expr::Literal(_)) =>
            {
                let new_right = right.clone().rewrite(self)?.data;
                let new_expr = BinaryExpr::new(left.clone(), *op, Box::new(new_right));
                Transformed::new(
                    Expr::BinaryExpr(new_expr),
                    true,
                    TreeNodeRecursion::Jump,
                )
            }

            // Leaf expr, create `new aggr expr` on it
            expr => {
                if matches!(expr, &Expr::AggregateFunction(_)) {
                    return internal_err!("found nested aggr function expr:{expr}");
                }

                let mut new_aggr_function = self.origin_aggr_function.clone();
                new_aggr_function.params.args = vec![expr.clone()];
                let new_aggr_expr = Expr::AggregateFunction(new_aggr_function);
                // We should also collect and dedup `new_aggr_expr`s, for create new`Aggregate` node later
                self.new_aggr_exprs.insert(new_aggr_expr.clone());
                Transformed::new(new_aggr_expr, true, TreeNodeRecursion::Jump)
            }
        };

        Ok(transformed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_optimized_plan_eq_snapshot;
    use crate::test::*;
    use crate::Optimizer;
    use crate::OptimizerContext;

    use arrow::datatypes::DataType;
    use datafusion_common::Result;
    use datafusion_expr::expr::ScalarFunction;
    use datafusion_expr::{
        cast, col, lit, ColumnarValue, LogicalPlanBuilder, ScalarFunctionArgs, ScalarUDF,
        ScalarUDFImpl, Signature, TypeSignature,
    };

    use datafusion_functions_aggregate::expr_fn::count;
    use datafusion_functions_aggregate::expr_fn::sum;

    use std::sync::Arc;

    macro_rules! assert_optimized_plan_equal {
            (
                $plan:expr,
                @ $expected:literal $(,)?
            ) => {{
                let rule: Arc<dyn crate::OptimizerRule + Send + Sync> = Arc::new(TransformLinearAggregation::new());
                assert_optimized_plan_eq_snapshot!(
                    rule,
                    $plan,
                    @ $expected,
                )
            }};
        }

    #[test]
    fn test_eliminate_gby_literal() {
        let scan = test_table_scan().unwrap();
        let plan = LogicalPlanBuilder::from(scan)
            .aggregate(
                Vec::<Expr>::new(),
                vec![
                    sum(cast(col("c"), DataType::Int64) + lit(1_i64)),
                    sum(cast(col("c"), DataType::Int64) + lit(2_i64)),
                    sum(cast(col("c"), DataType::Int64) + lit(3_i64)),
                ],
            )
            .unwrap()
            .build()
            .unwrap();

        let rule: Arc<dyn OptimizerRule + Send + Sync> =
            Arc::new(TransformLinearAggregation::new());
        let opt_context = OptimizerContext::new().with_max_passes(1);
        let optimizer = Optimizer::with_rules(vec![Arc::clone(&rule)]);
        let optimized_plan = optimizer.optimize(plan, &opt_context, |_, _| {}).unwrap();
        println!("{optimized_plan}");

        // assert_optimized_plan_equal!(plan, @r"
        // Projection: test.a, UInt32(1), count(test.c)
        //   Aggregate: groupBy=[[test.a]], aggr=[[count(test.c)]]
        //     TableScan: test
        // ")
    }

    //     #[test]
    //     fn test_eliminate_constant() -> Result<()> {
    //         let scan = test_table_scan()?;
    //         let plan = LogicalPlanBuilder::from(scan)
    //             .aggregate(vec![lit("test"), lit(123u32)], vec![count(col("c"))])?
    //             .build()?;

    //         assert_optimized_plan_equal!(plan, @r#"
    //         Projection: Utf8("test"), UInt32(123), count(test.c)
    //           Aggregate: groupBy=[[]], aggr=[[count(test.c)]]
    //             TableScan: test
    //         "#)
    //     }
}
