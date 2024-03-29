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

//! OptimizeProjections rule aims achieving the most effective use of projections
//! in plans. It ensures that query plans are free from unnecessary projections
//! and that no unused columns are propagated unnecessarily between plans.
//!
//! The rule is designed to enhance query performance by:
//! 1. Preventing the transfer of unused columns from leaves to root.
//! 2. Ensuring projections are used only when they contribute to narrowing the schema,
//!    or when necessary for evaluation or aliasing.
//!
//! The optimization is conducted in two phases:
//!
//! Top-down Phase:
//! ---------------
//! - Traverses the plan from root to leaves. If the node is:
//!   1. Projection node, it may:
//!      a. Merge it with its input projection if merge is beneficial.
//!      b. Remove the projection if it is redundant.
//!      c. Narrow the Projection if possible.
//!      d. The projection can be nested into the input.
//!      e. Do nothing, otherwise.
//!   2. Non-Projection node:
//!      a. Schema needs pruning: Insert the necessary projection into input.
//!      b. All fields are required: Do nothing.
//!
//! Bottom-up Phase (resides in with_new_children() implementation of ConcreteTreeNode):
//! ----------------
//! This pass is required because modifying a plan node can change the column
//! indices used by output nodes. When such a change occurs, we store the old
//! and new indices of the columns in the node's state. We then proceed from
//! the leaves to the root, updating the indices of columns in the plans by
//! referencing these mapping records. After the top-down phase, also some
//! unnecessary projections may emerge. When projections check its input schema
//! mapping, it can remove itself and assign new schema mapping to the new node,
//! which was the projection's input formerly.

use std::collections::{HashMap, HashSet};
use std::mem;
use std::sync::Arc;

use super::PhysicalOptimizerRule;
use crate::datasource::physical_plan::CsvExec;
use crate::error::Result;
use crate::physical_plan::filter::FilterExec;
use crate::physical_plan::projection::ProjectionExec;
use crate::physical_plan::ExecutionPlan;

use arrow_schema::SchemaRef;
use datafusion_common::config::ConfigOptions;
use datafusion_common::tree_node::{
    ConcreteTreeNode, Transformed, TreeNode, TreeNodeRecursion,
};
use datafusion_common::{JoinSide, JoinType};
use datafusion_physical_expr::expressions::{Column, Literal};
use datafusion_physical_expr::utils::collect_columns;
use datafusion_physical_expr::window::WindowExpr;
use datafusion_physical_expr::{
    AggregateExpr, LexOrdering, Partitioning, PhysicalExpr, PhysicalExprRef,
    PhysicalSortExpr,
};
use datafusion_physical_plan::aggregates::{
    AggregateExec, AggregateMode, PhysicalGroupBy,
};
use datafusion_physical_plan::coalesce_batches::CoalesceBatchesExec;
use datafusion_physical_plan::coalesce_partitions::CoalescePartitionsExec;
use datafusion_physical_plan::insert::FileSinkExec;
use datafusion_physical_plan::joins::utils::{
    ColumnIndex, JoinFilter, JoinOn, JoinOnRef,
};
use datafusion_physical_plan::joins::{
    CrossJoinExec, HashJoinExec, NestedLoopJoinExec, SortMergeJoinExec,
    SymmetricHashJoinExec,
};
use datafusion_physical_plan::limit::{GlobalLimitExec, LocalLimitExec};
use datafusion_physical_plan::repartition::RepartitionExec;
use datafusion_physical_plan::sorts::sort::SortExec;
use datafusion_physical_plan::sorts::sort_preserving_merge::SortPreservingMergeExec;
use datafusion_physical_plan::union::{InterleaveExec, UnionExec};
use datafusion_physical_plan::windows::{BoundedWindowAggExec, WindowAggExec};

use itertools::Itertools;

/// The tree node for the rule of [`OptimizeProjections`]. It stores the necessary
/// fields for column requirements and changed indices of columns.
#[derive(Debug, Clone)]
pub struct ProjectionOptimizer {
    /// The plan resides in the node
    pub plan: Arc<dyn ExecutionPlan>,
    /// The node above expects it can reach these columns.
    pub required_columns: HashSet<Column>,
    /// The nodes above will be updated according to these matches. First element indicates
    /// the initial column index, and the second element is for the updated version.
    pub schema_mapping: HashMap<Column, Column>,
    /// Children nodes
    pub children_nodes: Vec<ProjectionOptimizer>,
}

/// This type defines whether a column is required, in case of pairing with `true` value, or is
/// not required, in case of pairing with `false`. It is constructed based on output schema of a plan.
type ColumnRequirements = HashMap<Column, bool>;

impl ProjectionOptimizer {
    /// Constructs the empty tree according to the plan. All state information is empty initially.
    fn new_default(plan: Arc<dyn ExecutionPlan>) -> Self {
        let children = plan.children();
        Self {
            plan,
            required_columns: HashSet::new(),
            schema_mapping: HashMap::new(),
            children_nodes: children.into_iter().map(Self::new_default).collect(),
        }
    }

    /// Recursively called transform function while traversing from root node
    /// to leaf nodes. It only addresses the self and child node, and make
    /// the necessary changes on them, does not deep dive.
    fn adjust_node_with_requirements(mut self) -> Result<Self> {
        // If the node is a source provider, no need a change.
        if self.children_nodes.is_empty() {
            // We also clean the requirements, since we would like
            // to leave payload-free nodes after the rule finishes.
            self.required_columns.clear();
            return Ok(self);
        }

        if self.plan.as_any().is::<ProjectionExec>() {
            // If the node is a projection, it is analyzed and may be rewritten
            // to make the projection more efficient, or even it may be removed.
            self.optimize_projections()
        } else {
            // If the node is any other plan, a projection may be inserted to its input.
            self.try_projection_insertion()
        }
    }

    /// The function tries 4 cases:
    /// 1. If the input plan is also a projection, they can be merged into one projection.
    /// 2. The projection can be removed.
    /// 3. The projection can get narrower.
    /// 4. The projection can be embedded into the plan below.
    /// If none of them is possible, it remains unchanged.
    fn optimize_projections(mut self) -> Result<Self> {
        let projection_input = &self.plan.children()[0];

        // We first need to check having 2 sequential projections in case of merging them.
        if projection_input.as_any().is::<ProjectionExec>() {
            self = match self.try_unify_projections()? {
                unified_plans if unified_plans.transformed => {
                    // We need to re-run the rule on the new node since it may need further optimizations.
                    // There may be 3 sequential projections, or the unified node may also be removed or narrowed.
                    return unified_plans.data.optimize_projections();
                }
                no_change => no_change.data,
            };
        }

        // The projection can be removed. To avoid making unnecessary operations,
        // try_remove should be called before try_narrow.
        self = match self.try_remove_projection()? {
            removed if removed.transformed => {
                // We need to re-run the rule on the current node. It is
                // a new plan node and may need optimizations for sure.
                return removed.data.adjust_node_with_requirements();
            }
            no_change => no_change.data,
        };

        // The projection can get narrower.
        self = match self.try_narrow_projection()? {
            narrowed if narrowed.transformed => {
                // We need to re-run the rule on the new node since it may be removed within the plan below.
                return narrowed.data.optimize_projections();
            }
            no_change => no_change.data,
        };

        // HashJoinExec can own the projection above.
        if projection_input.as_any().is::<HashJoinExec>() {
            return match self.try_embed_to_hash_join()? {
                join if join.transformed => {
                    // Re-run on the new HashJoin node
                    return join.data.adjust_node_with_requirements();
                }
                projection => Ok(projection.data),
            };
        }

        // Source providers:
        if projection_input.as_any().is::<CsvExec>() {
            return self.try_projected_csv();
        }
        // TODO: Other source execs can be implemented here if projection can be applied inside them.

        // If none of them possible, we will continue to next node. Output requirements
        // of the projection in terms of projection input are inserted to child node.
        self.children_nodes[0].required_columns = if let Some(projection_plan) =
            self.plan.as_any().downcast_ref::<ProjectionExec>()
        {
            self.required_columns
                .iter()
                .flat_map(|e| collect_columns(&projection_plan.expr()[e.index()].0))
                .collect::<HashSet<_>>()
        } else {
            // If the method is used with a non-projection plan, we must sustain the execution safely.
            collect_columns_in_plan_schema(projection_input)
        };

        Ok(self)
    }

    /// If it is beneficial, unifies the projection and its input, which is also a [`ProjectionExec`].
    fn try_unify_projections(mut self) -> Result<Transformed<Self>> {
        // These are known to be a ProjectionExec.
        let Some(projection) = self.plan.as_any().downcast_ref::<ProjectionExec>() else {
            return Ok(Transformed::no(self));
        };
        let Some(child_projection) = self.children_nodes[0]
            .plan
            .as_any()
            .downcast_ref::<ProjectionExec>()
        else {
            return Ok(Transformed::no(self));
        };

        // Projection can be beneficial if it caches any computation which are used more than once.
        if caching_projections(projection, child_projection)? {
            return Ok(Transformed::no(self));
        }

        let mut projected_exprs = vec![];
        for (expr, alias) in projection.expr() {
            let Some(expr) =
                update_expr_with_projection(expr, child_projection.expr(), true)?
            else {
                return Ok(Transformed::no(self));
            };
            projected_exprs.push((expr, alias.clone()));
        }

        let new_plan =
            ProjectionExec::try_new(projected_exprs, child_projection.input().clone())
                .map(|e| Arc::new(e) as _)?;

        Ok(Transformed::yes(Self {
            plan: new_plan,
            // Schema of the projection does not change,
            // so no need any update on state variables.
            required_columns: self.required_columns,
            schema_mapping: self.schema_mapping,
            children_nodes: self.children_nodes.swap_remove(0).children_nodes,
        }))
    }

    /// Tries to remove the [`ProjectionExec`]. When these conditions are satisfied,
    /// the projection can be safely removed:
    /// 1. Projection must have all column expressions without aliases.
    /// 2. Projection input is fully required by the projection output requirements.
    fn try_remove_projection(mut self) -> Result<Transformed<Self>> {
        // It must be a projection.
        let Some(projection_exec) = self.plan.as_any().downcast_ref::<ProjectionExec>()
        else {
            return Ok(Transformed::no(self));
        };

        // The projection must have all column expressions without aliases.
        let Some(projection_columns) =
            try_collect_alias_free_columns(projection_exec.expr())
        else {
            return Ok(Transformed::no(self));
        };

        // Input requirements of the projection in terms of projection's parent requirements:
        let projection_requires =
            map_parent_reqs_to_input_reqs(&self.required_columns, &projection_columns);

        // If all fields of the input are necessary, we can remove the projection.
        let input_columns = collect_columns_in_plan_schema(projection_exec.input());
        if all_input_columns_required(&input_columns, &projection_requires) {
            let schema_mapping = index_changes_after_projection_removal(
                self.required_columns,
                &projection_columns,
            );
            let new_current_node = self.children_nodes.swap_remove(0);

            Ok(Transformed::yes(ProjectionOptimizer {
                plan: new_current_node.plan,
                required_columns: projection_requires,
                schema_mapping,
                children_nodes: new_current_node.children_nodes,
            }))
        } else {
            Ok(Transformed::no(self))
        }
    }

    /// Compares the inputs and outputs of the projection. If the projection can be
    /// rewritten with a narrower schema, it is done so. Otherwise, it returns `None`.
    fn try_narrow_projection(self) -> Result<Transformed<ProjectionOptimizer>> {
        // It must be a projection.
        let Some(projection_exec) = self.plan.as_any().downcast_ref::<ProjectionExec>()
        else {
            return Ok(Transformed::no(self));
        };

        let requirement_map = analyze_requirements(&self);
        let (used_columns, unused_columns) = split_column_requirements(&requirement_map);
        if unused_columns.is_empty() {
            // All projected items are used.
            return Ok(Transformed::no(self));
        }
        let projected_exprs = collect_used_columns(projection_exec.expr(), &used_columns);
        let mut new_mapping =
            calculate_column_mapping(&self.required_columns, &unused_columns);

        let new_projection_plan = Arc::new(ProjectionExec::try_new(
            projected_exprs,
            self.children_nodes[0].plan.clone(),
        )?);

        let new_projection_requires = self
            .required_columns
            .into_iter()
            .map(|col| new_mapping.remove(&col).unwrap_or(col))
            .collect();

        Ok(Transformed::yes(ProjectionOptimizer {
            plan: new_projection_plan,
            required_columns: new_projection_requires,
            schema_mapping: new_mapping,
            children_nodes: self.children_nodes,
        }))
    }

    /// By embedding projections to hash joins, we can reduce the cost of `build_batch_from_indices`
    /// in hash join (build_batch_from_indices need to can compute::take() for each column) and avoid
    /// unnecessary output creation.
    fn try_embed_to_hash_join(mut self) -> Result<Transformed<ProjectionOptimizer>> {
        let Some(projection) = self.plan.as_any().downcast_ref::<ProjectionExec>() else {
            return Ok(Transformed::no(self));
        };
        let Some(hash_join) = projection.input().as_any().downcast_ref::<HashJoinExec>()
        else {
            return Ok(Transformed::no(self));
        };

        // Collect all column indices from the given projection expressions.
        let projection_index = collect_column_indices_in_proj_exprs(projection.expr());

        // If the projection indices is the same as the input columns, we don't need to embed the projection to hash join.
        // Check the projection_index is 0..n-1 and the length of projection_index is the same as the length of hash_join schema fields.
        if projection_index.is_empty()
            || (projection_index.len() == projection_index.last().unwrap() + 1
                && projection_index.len() == hash_join.schema().fields().len())
        {
            self.children_nodes[0].required_columns =
                collect_columns_in_plan_schema(&self.children_nodes[0].plan);
            return Ok(Transformed::no(self));
        }

        let new_hash_join =
            Arc::new(hash_join.with_projection(Some(projection_index.clone()))?)
                as Arc<dyn ExecutionPlan>;

        // Build projection expressions for update_expr. Zip the projection_index with the new_hash_join output schema fields.
        let embed_project_exprs = projection_index
            .iter()
            .zip(new_hash_join.schema().fields())
            .map(|(index, field)| {
                (
                    Arc::new(Column::new(field.name(), *index)) as Arc<dyn PhysicalExpr>,
                    field.name().to_owned(),
                )
            })
            .collect::<Vec<_>>();

        let mut new_projection_exprs = Vec::with_capacity(projection.expr().len());
        for (expr, alias) in projection.expr() {
            // update column index for projection expression since the input schema has been changed.
            let Some(expr) =
                update_expr_with_projection(expr, embed_project_exprs.as_slice(), false)?
            else {
                self.children_nodes[0].required_columns =
                    collect_columns_in_plan_schema(&self.children_nodes[0].plan);
                return Ok(Transformed::no(self));
            };
            new_projection_exprs.push((expr, alias.clone()));
        }
        // Old projection may contain some alias or expression such as `a + 1` and `CAST('true' AS BOOLEAN)`,
        // but our projection_exprs in hash join just contain column, so we need to create the new projection
        // to keep the original projection.
        let new_projection =
            ProjectionExec::try_new(new_projection_exprs, new_hash_join.clone())?;

        if is_projection_removable(&new_projection) {
            let required_columns = collect_columns_in_plan_schema(&new_hash_join);
            Ok(Transformed::yes(Self {
                plan: new_hash_join,
                required_columns,
                schema_mapping: HashMap::new(),
                children_nodes: self.children_nodes.swap_remove(0).children_nodes,
            }))
        } else {
            let required_columns = collect_columns_in_plan_schema(&new_hash_join);
            let new_join_node = Self {
                plan: new_hash_join,
                required_columns,
                schema_mapping: HashMap::new(),
                children_nodes: self.children_nodes.swap_remove(0).children_nodes,
            };
            let plan = Arc::new(new_projection) as Arc<dyn ExecutionPlan>;
            Ok(Transformed::no(Self {
                plan,
                required_columns: HashSet::new(),
                schema_mapping: HashMap::new(),
                children_nodes: vec![new_join_node],
            }))
        }
    }

    /// Tries to embed [`ProjectionExec`] into its input [`CsvExec`].
    fn try_projected_csv(self) -> Result<ProjectionOptimizer> {
        // These plans are known.
        let Some(projection) = self.plan.as_any().downcast_ref::<ProjectionExec>() else {
            return Ok(self);
        };
        let Some(csv) = projection.input().as_any().downcast_ref::<CsvExec>() else {
            return Ok(self);
        };
        // If there is any non-column or alias-carrier expression, Projection should not be removed.
        // This process can be moved into CsvExec, but it could be a conflict of their responsibility.
        if let Some(projection_columns) =
            try_collect_alias_free_columns(projection.expr())
        {
            let mut file_scan = csv.base_config().clone();
            let new_projections = new_projections_for_columns(
                &projection_columns,
                &file_scan
                    .projection
                    .unwrap_or((0..csv.schema().fields().len()).collect()),
            );

            file_scan.projection = Some(new_projections);

            Ok(ProjectionOptimizer {
                plan: Arc::new(CsvExec::new(
                    file_scan,
                    csv.has_header(),
                    csv.delimiter(),
                    csv.quote(),
                    csv.escape(),
                    csv.file_compression_type,
                )) as _,
                required_columns: HashSet::new(),
                schema_mapping: HashMap::new(), // Sources cannot have a mapping.
                children_nodes: vec![],
            })
        } else {
            let mut file_scan = csv.base_config().clone();
            let required_columns =
                collect_column_indices_in_proj_exprs(projection.expr());
            let file_scan_projection = file_scan
                .projection
                .unwrap_or((0..csv.schema().fields().len()).collect());
            let (used_indices, unused_indices) =
                split_column_indices(file_scan_projection, required_columns);
            let new_exprs = update_proj_exprs(projection.expr(), unused_indices)?;
            file_scan.projection = Some(used_indices);
            let new_csv = ProjectionOptimizer {
                plan: Arc::new(CsvExec::new(
                    file_scan,
                    csv.has_header(),
                    csv.delimiter(),
                    csv.quote(),
                    csv.escape(),
                    csv.file_compression_type,
                )) as _,
                required_columns: HashSet::new(),
                schema_mapping: HashMap::new(), // Sources cannot have a mapping.
                children_nodes: vec![],
            };

            Ok(Self {
                plan: Arc::new(ProjectionExec::try_new(new_exprs, new_csv.plan.clone())?)
                    as Arc<dyn ExecutionPlan>,
                required_columns: self.required_columns,
                schema_mapping: self.schema_mapping,
                children_nodes: vec![new_csv],
            })
        }
    }

    /// If the node plan can be rewritten with a narrower schema, a projection is inserted
    /// into its input to do so. The node plans are rewritten according to its new input,
    /// and the mapping of old indices vs. new indices is put to node's related field.
    /// When this function returns and recursion on the node finishes, the upper node plans
    /// are rewritten according to this mapping. This function also updates the parent
    /// requirements and extends them with self requirements before inserting them to its child(ren).
    fn try_projection_insertion(mut self) -> Result<Self> {
        let plan = self.plan.clone();

        // These plans preserve the input schema, and do not add new requirements.
        if let Some(coal_batches) = plan.as_any().downcast_ref::<CoalesceBatchesExec>() {
            self = self.try_insert_below_coalesce_batches(coal_batches)?;
        } else if plan
            .as_any()
            .downcast_ref::<CoalescePartitionsExec>()
            .is_some()
        {
            self = self.try_insert_below_coalesce_partitions()?;
        } else if let Some(global_limit) = plan.as_any().downcast_ref::<GlobalLimitExec>()
        {
            self = self.try_insert_below_global_limit(global_limit)?;
        } else if let Some(local_limit) = plan.as_any().downcast_ref::<LocalLimitExec>() {
            self = self.try_insert_below_local_limit(local_limit)?;
        }
        // These plans also preserve the input schema, but may extend requirements.
        else if let Some(filter) = plan.as_any().downcast_ref::<FilterExec>() {
            self = self.try_insert_below_filter(filter)?;
        } else if let Some(repartition) = plan.as_any().downcast_ref::<RepartitionExec>()
        {
            self = self.try_insert_below_repartition(repartition)?;
        } else if let Some(sort) = plan.as_any().downcast_ref::<SortExec>() {
            self = self.try_insert_below_sort(sort)?;
        } else if let Some(sort_merge) =
            plan.as_any().downcast_ref::<SortPreservingMergeExec>()
        {
            self = self.try_insert_below_sort_preserving_merge(sort_merge)?;
        }
        // Preserves schema and do not change requirements, but have multi-child.
        else if plan.as_any().downcast_ref::<UnionExec>().is_some() {
            self = self.try_insert_below_union()?;
        } else if plan.as_any().downcast_ref::<InterleaveExec>().is_some() {
            self = self.try_insert_below_interleave()?;
        }
        // Concatenates schemas and do not change requirements.
        else if let Some(cj) = plan.as_any().downcast_ref::<CrossJoinExec>() {
            self = self.try_insert_below_cross_join(cj)?
        }
        // Joins and aggregations require special attention.
        else if let Some(hj) = plan.as_any().downcast_ref::<HashJoinExec>() {
            self = self.try_insert_below_hash_join(hj)?
        } else if let Some(nlj) = plan.as_any().downcast_ref::<NestedLoopJoinExec>() {
            self = self.try_insert_below_nested_loop_join(nlj)?
        } else if let Some(smj) = plan.as_any().downcast_ref::<SortMergeJoinExec>() {
            self = self.try_insert_below_sort_merge_join(smj)?
        } else if let Some(shj) = plan.as_any().downcast_ref::<SymmetricHashJoinExec>() {
            self = self.try_insert_below_symmetric_hash_join(shj)?
        } else if let Some(agg) = plan.as_any().downcast_ref::<AggregateExec>() {
            if !is_agg_expr_rewritable(agg.aggr_expr()) {
                self.children_nodes[0].required_columns =
                    collect_columns_in_plan_schema(&self.children_nodes[0].plan);
                return Ok(self);
            }
            self = self.try_insert_below_aggregate(agg)?
        } else if let Some(w_agg) = plan.as_any().downcast_ref::<WindowAggExec>() {
            if !is_window_expr_rewritable(w_agg.window_expr()) {
                self.children_nodes[0].required_columns =
                    collect_columns_in_plan_schema(&self.children_nodes[0].plan);
                return Ok(self);
            }
            self = self.try_insert_below_window_aggregate(w_agg)?
        } else if let Some(bw_agg) = plan.as_any().downcast_ref::<BoundedWindowAggExec>()
        {
            if !is_window_expr_rewritable(bw_agg.window_expr()) {
                self.children_nodes[0].required_columns =
                    collect_columns_in_plan_schema(&self.children_nodes[0].plan);
                return Ok(self);
            }
            self = self.try_insert_below_bounded_window_aggregate(bw_agg)?
        } else {
            self.children_nodes.iter_mut().for_each(|c| {
                c.required_columns = collect_columns_in_plan_schema(&c.plan)
            });
            return Ok(self);
        }
        Ok(self)
    }

    /// Attempts to insert a projection node below a `CoalesceBatchesExec` node.
    ///
    /// This method checks if there is any redundancy in the execution plan based on the requirements
    /// of the `CoalesceBatchesExec` node. If all columns are required, it updates the required columns of the child node.
    /// Otherwise, it inserts a new projection to optimize the plan.
    ///
    /// # Arguments
    /// * `coal_batches`: Reference to a `CoalesceBatchesExec` object.
    ///
    /// # Returns
    /// * `Result<ProjectionOptimizer>`: The modified `ProjectionOptimizer` after potentially inserting a projection.
    fn try_insert_below_coalesce_batches(
        mut self,
        coal_batches: &CoalesceBatchesExec,
    ) -> Result<ProjectionOptimizer> {
        // CoalesceBatchesExec does not change requirements. We can directly check whether there is a redundancy.
        let requirement_map = analyze_requirements(&self);
        if all_columns_required(&requirement_map) {
            self.children_nodes[0].required_columns =
                mem::take(&mut self.required_columns);
        } else {
            let (new_child, schema_mapping) = self.insert_projection(requirement_map)?;

            let plan = Arc::new(CoalesceBatchesExec::new(
                new_child.plan.clone(),
                coal_batches.target_batch_size(),
            )) as _;

            self = ProjectionOptimizer {
                plan,
                required_columns: HashSet::new(), // clear the requirements
                schema_mapping,
                children_nodes: vec![new_child],
            }
        }
        Ok(self)
    }

    /// Attempts to insert a projection node below a `CoalescePartitionsExec` node.
    ///
    /// Similar to `try_insert_below_coalesce_batches`, this method analyzes the requirements of the
    /// `CoalescePartitionsExec` node and modifies the execution plan accordingly.
    ///
    /// # Returns
    /// * `Result<ProjectionOptimizer>`: The potentially modified `ProjectionOptimizer`.
    fn try_insert_below_coalesce_partitions(mut self) -> Result<ProjectionOptimizer> {
        // CoalescePartitionsExec does not change requirements. We can directly check whether there is a redundancy.
        let requirement_map = analyze_requirements(&self);
        if all_columns_required(&requirement_map) {
            self.children_nodes[0].required_columns =
                mem::take(&mut self.required_columns);
        } else {
            let (new_child, schema_mapping) = self.insert_projection(requirement_map)?;
            let plan = Arc::new(CoalescePartitionsExec::new(new_child.plan.clone())) as _;

            self = ProjectionOptimizer {
                plan,
                required_columns: HashSet::new(), // clear the requirements
                schema_mapping,
                children_nodes: vec![new_child],
            }
        }
        Ok(self)
    }

    /// Attempts to insert a projection node below a `GlobalLimitExec` node.
    ///
    /// Analyzes the requirements imposed by `GlobalLimitExec` and optimizes the plan by potentially
    /// inserting a projection node to reduce the number of columns processed.
    ///
    /// # Arguments
    /// * `global_limit`: Reference to a `GlobalLimitExec` object.
    ///
    /// # Returns
    /// * `Result<ProjectionOptimizer>`: The modified `ProjectionOptimizer`.
    fn try_insert_below_global_limit(
        mut self,
        global_limit: &GlobalLimitExec,
    ) -> Result<ProjectionOptimizer> {
        // GlobalLimitExec does not change requirements. We can directly check whether there is a redundancy.
        let requirement_map = analyze_requirements(&self);
        if all_columns_required(&requirement_map) {
            // if all_columns_required(&requirement_map) {
            self.children_nodes[0].required_columns =
                mem::take(&mut self.required_columns);
        } else {
            let (new_child, schema_mapping) = self.insert_projection(requirement_map)?;
            let plan = Arc::new(GlobalLimitExec::new(
                new_child.plan.clone(),
                global_limit.skip(),
                global_limit.fetch(),
            )) as _;

            self = ProjectionOptimizer {
                plan,
                required_columns: HashSet::new(), // clear the requirements
                schema_mapping,
                children_nodes: vec![new_child],
            }
        }
        Ok(self)
    }

    /// Attempts to insert a projection node below a `LocalLimitExec` node.
    ///
    /// Optimizes the plan considering the requirements of `LocalLimitExec`, potentially inserting a projection node.
    ///
    /// # Arguments
    /// * `limit`: Reference to a `LocalLimitExec` object.
    ///
    /// # Returns
    /// * `Result<ProjectionOptimizer>`: The updated `ProjectionOptimizer`.
    fn try_insert_below_local_limit(
        mut self,
        local_limit: &LocalLimitExec,
    ) -> Result<ProjectionOptimizer> {
        // LocalLimitExec does not change requirements. We can directly check whether there is a redundancy.
        let requirement_map = analyze_requirements(&self);
        if all_columns_required(&requirement_map) {
            self.children_nodes[0].required_columns =
                mem::take(&mut self.required_columns);
        } else {
            let (new_child, schema_mapping) = self.insert_projection(requirement_map)?;
            let plan = Arc::new(LocalLimitExec::new(
                new_child.plan.clone(),
                local_limit.fetch(),
            )) as _;

            self = ProjectionOptimizer {
                plan,
                required_columns: HashSet::new(), // clear the requirements
                schema_mapping,
                children_nodes: vec![new_child],
            }
        }
        Ok(self)
    }

    /// Attempts to insert a projection node below a `FilterExec` node.
    ///
    /// Extends the required columns with those in the filter's predicate and optimizes the plan, potentially inserting
    /// a projection node.
    ///
    /// # Arguments
    /// * `filter`: Reference to a `FilterExec` object.
    ///
    /// # Returns
    /// * `Result<ProjectionOptimizer>`: The optimized `ProjectionOptimizer`.
    fn try_insert_below_filter(
        mut self,
        filter: &FilterExec,
    ) -> Result<ProjectionOptimizer> {
        // FilterExec extends the requirements with the columns in its predicate.
        self.required_columns
            .extend(collect_columns(filter.predicate()));

        let requirement_map = analyze_requirements(&self);
        if all_columns_required(&requirement_map) {
            self.children_nodes[0].required_columns =
                mem::take(&mut self.required_columns);
        } else {
            let (new_child, schema_mapping) = self.insert_projection(requirement_map)?;
            // Rewrite the predicate with possibly updated column indices.
            let new_predicate = update_column_index(filter.predicate(), &schema_mapping);
            let plan =
                Arc::new(FilterExec::try_new(new_predicate, new_child.plan.clone())?)
                    as _;

            self = ProjectionOptimizer {
                plan,
                required_columns: HashSet::new(), // clear the requirements
                schema_mapping,
                children_nodes: vec![new_child],
            }
        }
        Ok(self)
    }

    /// Attempts to insert a projection node below a `RepartitionExec` node.
    ///
    /// If `RepartitionExec` involves a hash repartition, it extends the requirements with the columns in the hashed expressions.
    /// The method then optimizes the execution plan accordingly.
    ///
    /// # Arguments
    /// * `repartition`: Reference to a `RepartitionExec` object.
    ///
    /// # Returns
    /// * `Result<ProjectionOptimizer>`: The potentially updated `ProjectionOptimizer`.
    fn try_insert_below_repartition(
        mut self,
        repartition: &RepartitionExec,
    ) -> Result<ProjectionOptimizer> {
        // If RepartitionExec applies a hash repartition, it extends
        // the requirements with the columns in the hashed expressions.
        if let Partitioning::Hash(exprs, _size) = repartition.partitioning() {
            self.required_columns
                .extend(exprs.iter().flat_map(collect_columns));
        }

        let requirement_map = analyze_requirements(&self);
        if all_columns_required(&requirement_map) {
            self.children_nodes[0].required_columns =
                mem::take(&mut self.required_columns);
        } else {
            let (new_child, schema_mapping) = self.insert_projection(requirement_map)?;
            // Rewrite the hashed expressions if there is any with possibly updated column indices.
            let new_partitioning = update_partitioning_expressions(
                repartition.partitioning(),
                &schema_mapping,
            );
            let plan = Arc::new(RepartitionExec::try_new(
                new_child.plan.clone(),
                new_partitioning,
            )?) as _;

            self = ProjectionOptimizer {
                plan,
                required_columns: HashSet::new(), // clear the requirements
                schema_mapping,
                children_nodes: vec![new_child],
            }
        }
        Ok(self)
    }

    /// Attempts to insert a projection node below a `SortExec` node.
    ///
    /// Extends the requirements with columns involved in the sort expressions and optimizes the execution plan,
    /// potentially inserting a projection node.
    ///
    /// # Arguments
    /// * `sort`: Reference to a `SortExec` object.
    ///
    /// # Returns
    /// * `Result<ProjectionOptimizer>`: The optimized `ProjectionOptimizer`.
    fn try_insert_below_sort(mut self, sort: &SortExec) -> Result<ProjectionOptimizer> {
        // SortExec extends the requirements with the columns in its sort expressions.
        self.required_columns.extend(
            sort.expr()
                .iter()
                .flat_map(|sort_expr| collect_columns(&sort_expr.expr)),
        );

        let requirement_map = analyze_requirements(&self);
        if all_columns_required(&requirement_map) {
            self.children_nodes[0].required_columns =
                mem::take(&mut self.required_columns);
        } else {
            let (new_child, schema_mapping) = self.insert_projection(requirement_map)?;
            // Rewrite the sort expressions with possibly updated column indices.
            let new_sort_exprs = update_sort_expressions(sort.expr(), &schema_mapping);
            let plan = Arc::new(
                SortExec::new(new_sort_exprs, new_child.plan.clone())
                    .with_preserve_partitioning(sort.preserve_partitioning())
                    .with_fetch(sort.fetch()),
            ) as _;

            self = ProjectionOptimizer {
                plan,
                required_columns: HashSet::new(), // clear the requirements
                schema_mapping,
                children_nodes: vec![new_child],
            }
        }
        Ok(self)
    }

    /// Attempts to insert a projection node below a `SortPreservingMergeExec` node.
    ///
    /// Similar to `try_insert_below_sort`, it extends the requirements with columns in the sort expressions and
    /// optimizes the plan accordingly.
    ///
    /// # Arguments
    /// * `sortp_merge`: Reference to a `SortPreservingMergeExec` object.
    ///
    /// # Returns
    /// * `Result<ProjectionOptimizer>`: The modified `ProjectionOptimizer`.
    fn try_insert_below_sort_preserving_merge(
        mut self,
        sort_merge: &SortPreservingMergeExec,
    ) -> Result<ProjectionOptimizer> {
        // SortPreservingMergeExec extends the requirements with the columns in its sort expressions.
        self.required_columns.extend(
            sort_merge
                .expr()
                .iter()
                .flat_map(|sort_expr| collect_columns(&sort_expr.expr)),
        );

        let requirement_map = analyze_requirements(&self);
        if all_columns_required(&requirement_map) {
            self.children_nodes[0].required_columns =
                mem::take(&mut self.required_columns);
        } else {
            let (new_child, schema_mapping) = self.insert_projection(requirement_map)?;
            // Rewrite the sort expressions with possibly updated column indices.
            let new_sort_exprs =
                update_sort_expressions(sort_merge.expr(), &schema_mapping);
            let plan = Arc::new(
                SortPreservingMergeExec::new(new_sort_exprs, new_child.plan.clone())
                    .with_fetch(sort_merge.fetch()),
            ) as _;

            self = ProjectionOptimizer {
                plan,
                required_columns: HashSet::new(), // clear the requirements
                schema_mapping,
                children_nodes: vec![new_child],
            }
        }
        Ok(self)
    }

    /// Attempts to insert a projection node below a `UnionExec` node in the query execution plan.
    ///
    /// This method checks the requirements of the current execution plan to determine if there is any redundancy
    /// when it comes to column usage in the context of a `UnionExec`. If all columns are required as per the
    /// requirement map, the method updates the required columns for all child nodes accordingly. If not all
    /// columns are required, it inserts new projection nodes to optimize the plan, leading to a more efficient
    /// execution by reducing unnecessary data processing.
    ///
    /// # Returns
    /// * `Result<ProjectionOptimizer>`: The optimized `ProjectionOptimizer` after potentially inserting projection nodes.
    fn try_insert_below_union(mut self) -> Result<ProjectionOptimizer> {
        // UnionExec does not change requirements. We can directly check whether there is a redundancy.
        let requirement_map = analyze_requirements(&self);
        if all_columns_required(&requirement_map) {
            let required_columns = mem::take(&mut self.required_columns);
            self.children_nodes
                .iter_mut()
                .for_each(|c| c.required_columns = required_columns.clone());
        } else {
            let (new_children, schema_mapping) =
                self.insert_multi_projection_below_union(requirement_map)?;
            let plan = Arc::new(UnionExec::new(
                new_children.iter().map(|c| c.plan.clone()).collect(),
            )) as _;

            self = ProjectionOptimizer {
                plan,
                required_columns: HashSet::new(), // clear the requirements
                schema_mapping,
                children_nodes: new_children,
            }
        }
        Ok(self)
    }

    /// Attempts to insert a projection node below an `InterleaveExec` node in the query execution plan.
    ///
    /// Similar to `try_insert_below_union`, this method analyzes the requirements of the `InterleaveExec` node and
    /// modifies the execution plan accordingly. If all columns are required, it updates the required columns for
    /// each child node. Otherwise, it inserts new projection nodes for optimization. This process can lead to a
    /// more efficient execution by minimizing the data processed in the context of interleaved execution.
    ///
    /// # Returns
    /// * `Result<ProjectionOptimizer>`: The potentially modified `ProjectionOptimizer` after the optimization process.
    fn try_insert_below_interleave(mut self) -> Result<ProjectionOptimizer> {
        let requirement_map = analyze_requirements(&self);
        if all_columns_required(&requirement_map) {
            let required_columns = mem::take(&mut self.required_columns);
            self.children_nodes
                .iter_mut()
                .for_each(|c| c.required_columns = required_columns.clone());
        } else {
            let (new_children, schema_mapping) =
                self.insert_multi_projection_below_union(requirement_map)?;
            let plan = Arc::new(InterleaveExec::try_new(
                new_children.iter().map(|c| c.plan.clone()).collect(),
            )?) as _;

            self = ProjectionOptimizer {
                plan,
                required_columns: HashSet::new(), // clear the requirements
                schema_mapping,
                children_nodes: new_children,
            }
        }
        Ok(self)
    }

    /// Attempts to insert projection nodes below a `CrossJoinExec` node in the query execution plan.
    ///
    /// This method first analyzes the requirements for both the left and right sides of the cross join. Depending on these
    /// requirements, it may insert projections on either or both sides of the join. Specifically, if not all columns are
    /// required on either side, it inserts the necessary projection nodes to streamline the join operation. If all columns
    /// are required from both sides, it updates the required columns accordingly without adding any projections. This
    /// optimization is crucial for reducing the computational overhead in cross join operations.
    ///
    /// # Arguments
    /// * `cj`: Reference to the `CrossJoinExec` node in the plan.
    ///
    /// # Returns
    /// * `Result<ProjectionOptimizer>`: The updated `ProjectionOptimizer` after potentially inserting projection nodes.
    fn try_insert_below_cross_join(
        mut self,
        cj: &CrossJoinExec,
    ) -> Result<ProjectionOptimizer> {
        let left_size = cj.left().schema().fields().len();
        // CrossJoinExec does not add new requirements.
        let (analyzed_join_left, analyzed_join_right) = analyze_requirements_of_joins(
            cj.left(),
            cj.right(),
            &self.required_columns,
            left_size,
        );
        match (
            all_columns_required(&analyzed_join_left),
            all_columns_required(&analyzed_join_right),
        ) {
            // We need two projections on top of both children.
            (false, false) => {
                let (new_left_child, new_right_child, schema_mapping) = self
                    .insert_multi_projections_below_join(
                        left_size,
                        analyzed_join_left,
                        analyzed_join_right,
                    )?;
                let plan = Arc::new(CrossJoinExec::new(
                    new_left_child.plan.clone(),
                    new_right_child.plan.clone(),
                )) as _;

                self = ProjectionOptimizer {
                    plan,
                    required_columns: HashSet::new(),
                    schema_mapping,
                    children_nodes: vec![new_left_child, new_right_child],
                }
            }
            // Left child needs a projection.
            (false, true) => {
                let required_columns = mem::take(&mut self.required_columns);
                let mut right_child = self.children_nodes.swap_remove(1);

                let (new_left_child, mut left_schema_mapping) =
                    self.insert_projection_below_left_child(analyzed_join_left)?;

                right_child.required_columns =
                    update_right_child_requirements(&required_columns, left_size);

                let plan = Arc::new(CrossJoinExec::new(
                    new_left_child.plan.clone(),
                    right_child.plan.clone(),
                )) as _;
                let new_left_size = new_left_child.plan.schema().fields().len();
                left_schema_mapping = extend_left_mapping_with_right(
                    left_schema_mapping,
                    &right_child.plan,
                    left_size,
                    new_left_size,
                );
                self = ProjectionOptimizer {
                    plan,
                    required_columns: HashSet::new(),
                    schema_mapping: left_schema_mapping,
                    children_nodes: vec![new_left_child, right_child],
                }
            }
            // Right child needs a projection.
            (true, false) => {
                let required_columns = mem::take(&mut self.required_columns);
                let mut left_child = self.children_nodes.swap_remove(0);
                let (new_right_child, mut right_schema_mapping) =
                    self.insert_projection_below_right_child(analyzed_join_right)?;

                right_schema_mapping =
                    update_right_mapping(right_schema_mapping, left_size);

                left_child.required_columns =
                    collect_left_used_columns(required_columns, left_size);

                let plan = Arc::new(CrossJoinExec::new(
                    left_child.plan.clone(),
                    new_right_child.plan.clone(),
                )) as _;

                self = ProjectionOptimizer {
                    plan,
                    required_columns: HashSet::new(),
                    schema_mapping: right_schema_mapping,
                    children_nodes: vec![left_child, new_right_child],
                }
            }
            // All columns are required.
            (true, true) => {
                self.required_columns = HashSet::new();
                self.children_nodes.iter_mut().for_each(|c| {
                    c.required_columns = collect_columns_in_plan_schema(&c.plan);
                })
            }
        }
        Ok(self)
    }

    /// Attempts to insert a projection below a HashJoinExec node in a query plan.
    ///
    /// This function modifies the projection optimizer by analyzing and potentially
    /// inserting new projections below the HashJoinExec node based on the join conditions
    /// and the required columns in the query. The process involves analyzing both left
    /// and right sides of the join, updating equivalence and non-equivalence conditions,
    /// and reorganizing the required columns as needed. This function supports various
    /// join types, including Inner, Left, Right, Full, LeftAnti, LeftSemi, RightAnti,
    /// and RightSemi joins.
    ///
    /// # Arguments
    ///
    /// * `hj`: Reference to a HashJoinExec node representing the join operation in the query plan.
    ///
    /// # Returns
    ///
    /// Returns `Result<ProjectionOptimizer>`, which is an updated ProjectionOptimizer
    /// instance after potentially adding projections below the HashJoinExec node.
    /// On success, it contains the modified ProjectionOptimizer with new projections
    /// and updated plan. On failure, it returns an error indicating the issue encountered
    /// during the operation.
    fn try_insert_below_hash_join(
        mut self,
        hj: &HashJoinExec,
    ) -> Result<ProjectionOptimizer> {
        match hj.join_type() {
            JoinType::Inner | JoinType::Left | JoinType::Right | JoinType::Full => {
                let join_left_input_size = hj.left().schema().fields().len();
                let join_projection = hj
                    .projection
                    .clone()
                    .unwrap_or((0..hj.schema().fields().len()).collect());

                let hj_left_requirements = collect_hj_left_requirements(
                    &self.required_columns,
                    &join_projection,
                    join_left_input_size,
                    hj.left().schema(),
                    hj.on(),
                    hj.filter(),
                );

                let hj_right_requirements = collect_hj_right_requirements(
                    &self.required_columns,
                    &join_projection,
                    join_left_input_size,
                    hj.right().schema(),
                    hj.on(),
                    hj.filter(),
                );

                let left_input_columns = collect_columns_in_plan_schema(hj.left());
                let keep_left_same = left_input_columns.iter().all(|left_input_column| {
                    hj_left_requirements.contains(left_input_column)
                });

                let right_input_columns = collect_columns_in_plan_schema(hj.right());
                let keep_right_same =
                    right_input_columns.iter().all(|right_input_column| {
                        hj_right_requirements.contains(right_input_column)
                    });

                match (keep_left_same, keep_right_same) {
                    (false, false) => {
                        let (new_left_node, new_right_node) = update_hj_children(
                            &hj_left_requirements,
                            &hj_right_requirements,
                            self.children_nodes,
                            hj,
                        )?;

                        let (left_mapping, right_mapping) = update_hj_children_mapping(
                            &hj_left_requirements,
                            &hj_right_requirements,
                        );

                        let new_on =
                            update_join_on(hj.on(), &left_mapping, &right_mapping);
                        let new_filter =
                            rewrite_hj_filter(hj.filter(), &left_mapping, &right_mapping);
                        let new_projection = update_hj_projection(
                            hj.projection.clone(),
                            hj.left().schema(),
                            hj_left_requirements,
                            left_mapping,
                            right_mapping,
                            join_left_input_size,
                        );

                        let new_hash_join = HashJoinExec::try_new(
                            new_left_node.plan.clone(),
                            new_right_node.plan.clone(),
                            new_on,
                            new_filter,
                            hj.join_type(),
                            new_projection,
                            *hj.partition_mode(),
                            hj.null_equals_null(),
                        )?;
                        Ok(ProjectionOptimizer {
                            plan: Arc::new(new_hash_join),
                            required_columns: HashSet::new(),
                            schema_mapping: HashMap::new(),
                            children_nodes: vec![new_left_node, new_right_node],
                        })
                    }
                    (false, true) => {
                        let (new_left_node, right_node) = update_hj_left_child(
                            &hj_left_requirements,
                            &hj_right_requirements,
                            self.children_nodes,
                            hj,
                        )?;

                        let (left_mapping, right_mapping) = update_hj_children_mapping(
                            &hj_left_requirements,
                            &hj_right_requirements,
                        );

                        let new_on =
                            update_join_on(hj.on(), &left_mapping, &right_mapping);
                        let new_filter =
                            rewrite_hj_filter(hj.filter(), &left_mapping, &right_mapping);
                        let new_projection = update_hj_projection(
                            hj.projection.clone(),
                            hj.left().schema(),
                            hj_left_requirements,
                            left_mapping,
                            right_mapping,
                            join_left_input_size,
                        );

                        let new_hash_join = HashJoinExec::try_new(
                            new_left_node.plan.clone(),
                            right_node.plan.clone(),
                            new_on,
                            new_filter,
                            hj.join_type(),
                            new_projection,
                            *hj.partition_mode(),
                            hj.null_equals_null(),
                        )?;

                        Ok(ProjectionOptimizer {
                            plan: Arc::new(new_hash_join),
                            required_columns: HashSet::new(),
                            schema_mapping: HashMap::new(),
                            children_nodes: vec![new_left_node, right_node],
                        })
                    }
                    (true, false) => {
                        let (left_node, new_right_node) = update_hj_right_child(
                            &hj_left_requirements,
                            &hj_right_requirements,
                            self.children_nodes,
                            hj,
                        )?;

                        let (left_mapping, right_mapping) = update_hj_children_mapping(
                            &hj_left_requirements,
                            &hj_right_requirements,
                        );

                        let new_on =
                            update_join_on(hj.on(), &left_mapping, &right_mapping);
                        let new_filter =
                            rewrite_hj_filter(hj.filter(), &left_mapping, &right_mapping);
                        let new_projection = update_hj_projection(
                            hj.projection.clone(),
                            hj.left().schema(),
                            hj_left_requirements,
                            left_mapping,
                            right_mapping,
                            join_left_input_size,
                        );

                        let new_hash_join = HashJoinExec::try_new(
                            left_node.plan.clone(),
                            new_right_node.plan.clone(),
                            new_on,
                            new_filter,
                            hj.join_type(),
                            new_projection,
                            *hj.partition_mode(),
                            hj.null_equals_null(),
                        )?;

                        Ok(ProjectionOptimizer {
                            plan: Arc::new(new_hash_join),
                            required_columns: HashSet::new(),
                            schema_mapping: HashMap::new(),
                            children_nodes: vec![left_node, new_right_node],
                        })
                    }
                    (true, true) => {
                        self.required_columns = HashSet::new();
                        self.children_nodes.iter_mut().for_each(|c| {
                            c.required_columns = collect_columns_in_plan_schema(&c.plan);
                        });
                        Ok(self)
                    }
                }
            }
            JoinType::LeftSemi | JoinType::LeftAnti => {
                let join_left_input_size = hj.left().schema().fields().len();
                let join_projection = hj
                    .projection
                    .clone()
                    .unwrap_or((0..hj.left().schema().fields().len()).collect());

                let hj_left_requirements = collect_hj_left_requirements(
                    &self.required_columns,
                    &join_projection,
                    join_left_input_size,
                    hj.left().schema(),
                    hj.on(),
                    hj.filter(),
                );

                let hj_right_requirements = collect_hj_right_requirements(
                    &self.required_columns,
                    &join_projection,
                    join_left_input_size,
                    hj.right().schema(),
                    hj.on(),
                    hj.filter(),
                );

                let left_input_columns = collect_columns_in_plan_schema(hj.left());
                let keep_left_same = left_input_columns.iter().all(|left_input_column| {
                    hj_left_requirements.contains(left_input_column)
                });

                let right_input_columns = collect_columns_in_plan_schema(hj.right());
                let keep_right_same =
                    right_input_columns.iter().all(|right_input_column| {
                        hj_right_requirements.contains(right_input_column)
                    });

                match (keep_left_same, keep_right_same) {
                    (false, false) => {
                        let (new_left_node, new_right_node) = update_hj_children(
                            &hj_left_requirements,
                            &hj_right_requirements,
                            self.children_nodes,
                            hj,
                        )?;

                        let (left_mapping, right_mapping) = update_hj_children_mapping(
                            &hj_left_requirements,
                            &hj_right_requirements,
                        );

                        let new_on =
                            update_join_on(hj.on(), &left_mapping, &right_mapping);
                        let new_filter =
                            rewrite_hj_filter(hj.filter(), &left_mapping, &right_mapping);

                        let new_projection = update_hj_projection(
                            hj.projection.clone(),
                            hj.left().schema(),
                            hj_left_requirements,
                            left_mapping,
                            right_mapping,
                            join_left_input_size,
                        );

                        let new_hash_join = HashJoinExec::try_new(
                            new_left_node.plan.clone(),
                            new_right_node.plan.clone(),
                            new_on,
                            new_filter,
                            hj.join_type(),
                            new_projection,
                            *hj.partition_mode(),
                            hj.null_equals_null(),
                        )?;
                        Ok(ProjectionOptimizer {
                            plan: Arc::new(new_hash_join),
                            required_columns: HashSet::new(),
                            schema_mapping: HashMap::new(),
                            children_nodes: vec![new_left_node, new_right_node],
                        })
                    }
                    (false, true) => {
                        let (new_left_node, right_node) = update_hj_left_child(
                            &hj_left_requirements,
                            &hj_right_requirements,
                            self.children_nodes,
                            hj,
                        )?;

                        let (left_mapping, right_mapping) = update_hj_children_mapping(
                            &hj_left_requirements,
                            &hj_right_requirements,
                        );

                        let new_on =
                            update_join_on(hj.on(), &left_mapping, &right_mapping);
                        let new_filter =
                            rewrite_hj_filter(hj.filter(), &left_mapping, &right_mapping);
                        let new_projection = update_hj_projection(
                            hj.projection.clone(),
                            hj.left().schema(),
                            hj_left_requirements,
                            left_mapping,
                            right_mapping,
                            join_left_input_size,
                        );

                        let new_hash_join = HashJoinExec::try_new(
                            new_left_node.plan.clone(),
                            right_node.plan.clone(),
                            new_on,
                            new_filter,
                            hj.join_type(),
                            new_projection,
                            *hj.partition_mode(),
                            hj.null_equals_null(),
                        )?;

                        Ok(ProjectionOptimizer {
                            plan: Arc::new(new_hash_join),
                            required_columns: HashSet::new(),
                            schema_mapping: HashMap::new(),
                            children_nodes: vec![new_left_node, right_node],
                        })
                    }
                    (true, false) => {
                        let (left_node, new_right_node) = update_hj_right_child(
                            &hj_left_requirements,
                            &hj_right_requirements,
                            self.children_nodes,
                            hj,
                        )?;

                        let (left_mapping, right_mapping) = update_hj_children_mapping(
                            &hj_left_requirements,
                            &hj_right_requirements,
                        );

                        let new_on =
                            update_join_on(hj.on(), &left_mapping, &right_mapping);
                        let new_filter =
                            rewrite_hj_filter(hj.filter(), &left_mapping, &right_mapping);
                        let new_projection = update_hj_projection(
                            hj.projection.clone(),
                            hj.left().schema(),
                            hj_left_requirements,
                            left_mapping,
                            right_mapping,
                            join_left_input_size,
                        );

                        let new_hash_join = HashJoinExec::try_new(
                            left_node.plan.clone(),
                            new_right_node.plan.clone(),
                            new_on,
                            new_filter,
                            hj.join_type(),
                            new_projection,
                            *hj.partition_mode(),
                            hj.null_equals_null(),
                        )?;

                        Ok(ProjectionOptimizer {
                            plan: Arc::new(new_hash_join),
                            required_columns: HashSet::new(),
                            schema_mapping: HashMap::new(),
                            children_nodes: vec![left_node, new_right_node],
                        })
                    }
                    (true, true) => {
                        self.required_columns = HashSet::new();
                        self.children_nodes.iter_mut().for_each(|c| {
                            c.required_columns = collect_columns_in_plan_schema(&c.plan);
                        });
                        Ok(self)
                    }
                }
            }
            JoinType::RightSemi | JoinType::RightAnti => {
                let join_projection = hj
                    .projection
                    .clone()
                    .unwrap_or((0..hj.right().schema().fields().len()).collect());

                let hj_left_requirements = collect_right_hj_left_requirements(
                    hj.left().schema(),
                    hj.on(),
                    hj.filter(),
                );

                let hj_right_requirements = collect_right_hj_right_requirements(
                    &self.required_columns,
                    &join_projection,
                    hj.right().schema(),
                    hj.on(),
                    hj.filter(),
                );

                let left_input_columns = collect_columns_in_plan_schema(hj.left());
                let keep_left_same = left_input_columns.iter().all(|left_input_column| {
                    hj_left_requirements.contains(left_input_column)
                });

                let right_input_columns = collect_columns_in_plan_schema(hj.right());
                let keep_right_same =
                    right_input_columns.iter().all(|right_input_column| {
                        hj_right_requirements.contains(right_input_column)
                    });

                match (keep_left_same, keep_right_same) {
                    (false, false) => {
                        let (new_left_node, new_right_node) = update_hj_children(
                            &hj_left_requirements,
                            &hj_right_requirements,
                            self.children_nodes,
                            hj,
                        )?;

                        let (left_mapping, right_mapping) = update_hj_children_mapping(
                            &hj_left_requirements,
                            &hj_right_requirements,
                        );

                        let new_on =
                            update_join_on(hj.on(), &left_mapping, &right_mapping);
                        let new_filter =
                            rewrite_hj_filter(hj.filter(), &left_mapping, &right_mapping);

                        let new_projection = update_hj_projection_right(
                            hj.projection.clone(),
                            right_mapping,
                        );

                        let new_hash_join = HashJoinExec::try_new(
                            new_left_node.plan.clone(),
                            new_right_node.plan.clone(),
                            new_on,
                            new_filter,
                            hj.join_type(),
                            new_projection,
                            *hj.partition_mode(),
                            hj.null_equals_null(),
                        )?;
                        Ok(ProjectionOptimizer {
                            plan: Arc::new(new_hash_join),
                            required_columns: HashSet::new(),
                            schema_mapping: HashMap::new(),
                            children_nodes: vec![new_left_node, new_right_node],
                        })
                    }
                    (false, true) => {
                        let (new_left_node, right_node) = update_hj_left_child(
                            &hj_left_requirements,
                            &hj_right_requirements,
                            self.children_nodes,
                            hj,
                        )?;

                        let (left_mapping, right_mapping) = update_hj_children_mapping(
                            &hj_left_requirements,
                            &hj_right_requirements,
                        );

                        let new_on =
                            update_join_on(hj.on(), &left_mapping, &right_mapping);
                        let new_filter =
                            rewrite_hj_filter(hj.filter(), &left_mapping, &right_mapping);
                        let new_projection = update_hj_projection_right(
                            hj.projection.clone(),
                            left_mapping,
                        );

                        let new_hash_join = HashJoinExec::try_new(
                            new_left_node.plan.clone(),
                            right_node.plan.clone(),
                            new_on,
                            new_filter,
                            hj.join_type(),
                            new_projection,
                            *hj.partition_mode(),
                            hj.null_equals_null(),
                        )?;

                        Ok(ProjectionOptimizer {
                            plan: Arc::new(new_hash_join),
                            required_columns: HashSet::new(),
                            schema_mapping: HashMap::new(),
                            children_nodes: vec![new_left_node, right_node],
                        })
                    }
                    (true, false) => {
                        let (left_node, new_right_node) = update_hj_right_child(
                            &hj_left_requirements,
                            &hj_right_requirements,
                            self.children_nodes,
                            hj,
                        )?;

                        let (left_mapping, right_mapping) = update_hj_children_mapping(
                            &hj_left_requirements,
                            &hj_right_requirements,
                        );

                        let new_on =
                            update_join_on(hj.on(), &left_mapping, &right_mapping);
                        let new_filter =
                            rewrite_hj_filter(hj.filter(), &left_mapping, &right_mapping);
                        let new_projection = update_hj_projection_right(
                            hj.projection.clone(),
                            right_mapping,
                        );

                        let new_hash_join = HashJoinExec::try_new(
                            left_node.plan.clone(),
                            new_right_node.plan.clone(),
                            new_on,
                            new_filter,
                            hj.join_type(),
                            new_projection,
                            *hj.partition_mode(),
                            hj.null_equals_null(),
                        )?;

                        Ok(ProjectionOptimizer {
                            plan: Arc::new(new_hash_join),
                            required_columns: HashSet::new(),
                            schema_mapping: HashMap::new(),
                            children_nodes: vec![left_node, new_right_node],
                        })
                    }
                    (true, true) => {
                        self.required_columns = HashSet::new();
                        self.children_nodes.iter_mut().for_each(|c| {
                            c.required_columns = collect_columns_in_plan_schema(&c.plan);
                        });
                        Ok(self)
                    }
                }
            }
        }
    }

    fn try_insert_below_nested_loop_join(
        mut self,
        nlj: &NestedLoopJoinExec,
    ) -> Result<ProjectionOptimizer> {
        let left_size = nlj.left().schema().fields().len();
        // NestedLoopJoinExec extends the requirements with the columns in its equivalence and non-equivalence conditions.
        match nlj.join_type() {
            JoinType::RightAnti | JoinType::RightSemi => {
                self.required_columns =
                    update_right_requirements(self.required_columns, left_size);
            }
            _ => {}
        }
        self.required_columns
            .extend(collect_columns_in_join_conditions(
                &[],
                nlj.filter(),
                left_size,
                self.children_nodes[0].plan.schema(),
                self.children_nodes[1].plan.schema(),
            ));
        let (analyzed_join_left, analyzed_join_right) = analyze_requirements_of_joins(
            nlj.left(),
            nlj.right(),
            &self.required_columns,
            left_size,
        );

        match nlj.join_type() {
            JoinType::Inner | JoinType::Left | JoinType::Right | JoinType::Full => {
                match (
                    all_columns_required(&analyzed_join_left),
                    all_columns_required(&analyzed_join_right),
                ) {
                    // We need two projections on top of both children.
                    (false, false) => {
                        let new_filter = update_non_equivalence_conditions(
                            nlj.filter(),
                            &analyzed_join_left,
                            &analyzed_join_right,
                        );
                        let (new_left_child, new_right_child, schema_mapping) = self
                            .insert_multi_projections_below_join(
                                left_size,
                                analyzed_join_left,
                                analyzed_join_right,
                            )?;
                        let plan = Arc::new(NestedLoopJoinExec::try_new(
                            new_left_child.plan.clone(),
                            new_right_child.plan.clone(),
                            new_filter,
                            nlj.join_type(),
                        )?) as _;

                        self = ProjectionOptimizer {
                            plan,
                            required_columns: HashSet::new(),
                            schema_mapping,
                            children_nodes: vec![new_left_child, new_right_child],
                        }
                    }
                    (false, true) => {
                        let required_columns = mem::take(&mut self.required_columns);
                        let mut right_child = self.children_nodes.swap_remove(1);
                        let new_filter = update_non_equivalence_conditions(
                            nlj.filter(),
                            &analyzed_join_right,
                            &HashMap::new(),
                        );
                        let (new_left_child, mut left_schema_mapping) =
                            self.insert_projection_below_left_child(analyzed_join_left)?;

                        right_child.required_columns =
                            update_right_child_requirements(&required_columns, left_size);

                        let plan = Arc::new(NestedLoopJoinExec::try_new(
                            new_left_child.plan.clone(),
                            right_child.plan.clone(),
                            new_filter,
                            nlj.join_type(),
                        )?) as _;
                        let new_left_size = new_left_child.plan.schema().fields().len();
                        left_schema_mapping = extend_left_mapping_with_right(
                            left_schema_mapping,
                            &right_child.plan,
                            left_size,
                            new_left_size,
                        );

                        self = ProjectionOptimizer {
                            plan,
                            required_columns: HashSet::new(),
                            schema_mapping: left_schema_mapping,
                            children_nodes: vec![new_left_child, right_child],
                        }
                    }
                    (true, false) => {
                        let required_columns = mem::take(&mut self.required_columns);
                        let mut left_child = self.children_nodes.swap_remove(0);
                        let new_filter = update_non_equivalence_conditions(
                            nlj.filter(),
                            &HashMap::new(),
                            &analyzed_join_right,
                        );
                        let (new_right_child, mut right_schema_mapping) = self
                            .insert_projection_below_right_child(analyzed_join_right)?;

                        right_schema_mapping =
                            update_right_mapping(right_schema_mapping, left_size);

                        left_child.required_columns =
                            collect_left_used_columns(required_columns, left_size);

                        let plan = Arc::new(NestedLoopJoinExec::try_new(
                            left_child.plan.clone(),
                            new_right_child.plan.clone(),
                            new_filter,
                            nlj.join_type(),
                        )?) as _;

                        self = ProjectionOptimizer {
                            plan,
                            required_columns: HashSet::new(),
                            schema_mapping: right_schema_mapping,
                            children_nodes: vec![left_child, new_right_child],
                        }
                    }
                    // All columns are required.
                    (true, true) => {
                        self.required_columns = HashSet::new();
                        self.children_nodes.iter_mut().for_each(|c| {
                            c.required_columns = collect_columns_in_plan_schema(&c.plan);
                        })
                    }
                }
            }
            JoinType::LeftAnti | JoinType::LeftSemi => {
                match all_columns_required(&analyzed_join_left) {
                    false => {
                        let mut right_child = self.children_nodes.swap_remove(1);
                        let new_filter = update_non_equivalence_conditions(
                            nlj.filter(),
                            &analyzed_join_left,
                            &HashMap::new(),
                        );
                        let (new_left_child, left_schema_mapping) =
                            self.insert_projection_below_left_child(analyzed_join_left)?;

                        let plan = Arc::new(NestedLoopJoinExec::try_new(
                            new_left_child.plan.clone(),
                            right_child.plan.clone(),
                            new_filter,
                            nlj.join_type(),
                        )?) as _;

                        right_child.required_columns = analyzed_join_right
                            .into_iter()
                            .filter_map(
                                |(column, used)| if used { Some(column) } else { None },
                            )
                            .collect();
                        self = ProjectionOptimizer {
                            plan,
                            required_columns: HashSet::new(),
                            schema_mapping: left_schema_mapping,
                            children_nodes: vec![new_left_child, right_child],
                        }
                    }
                    true => {
                        self.children_nodes[0].required_columns =
                            collect_columns_in_plan_schema(&self.children_nodes[0].plan);
                        self.children_nodes[1].required_columns = analyzed_join_right
                            .into_iter()
                            .filter_map(
                                |(column, used)| if used { Some(column) } else { None },
                            )
                            .collect()
                    }
                }
            }
            JoinType::RightAnti | JoinType::RightSemi => {
                match all_columns_required(&analyzed_join_right) {
                    false => {
                        let mut left_child = self.children_nodes.swap_remove(0);
                        let new_filter = update_non_equivalence_conditions(
                            nlj.filter(),
                            &HashMap::new(),
                            &analyzed_join_right,
                        );
                        let (new_right_child, right_schema_mapping) = self
                            .insert_projection_below_right_child(analyzed_join_right)?;

                        let plan = Arc::new(NestedLoopJoinExec::try_new(
                            left_child.plan.clone(),
                            new_right_child.plan.clone(),
                            new_filter,
                            nlj.join_type(),
                        )?) as _;

                        left_child.required_columns = analyzed_join_left
                            .into_iter()
                            .filter_map(
                                |(column, used)| if used { Some(column) } else { None },
                            )
                            .collect();
                        self = ProjectionOptimizer {
                            plan,
                            required_columns: HashSet::new(),
                            schema_mapping: right_schema_mapping,
                            children_nodes: vec![left_child, new_right_child],
                        }
                    }
                    true => {
                        self.children_nodes[0].required_columns = analyzed_join_left
                            .into_iter()
                            .filter_map(
                                |(column, used)| if used { Some(column) } else { None },
                            )
                            .collect();
                        self.children_nodes[1].required_columns =
                            collect_columns_in_plan_schema(&self.children_nodes[1].plan);
                    }
                }
            }
        }
        Ok(self)
    }

    fn try_insert_below_sort_merge_join(
        mut self,
        smj: &SortMergeJoinExec,
    ) -> Result<ProjectionOptimizer> {
        let left_size = smj.left().schema().fields().len();
        // SortMergeJoin extends the requirements with the columns in its equivalence and non-equivalence conditions.
        match smj.join_type() {
            JoinType::RightAnti | JoinType::RightSemi => {
                self.required_columns =
                    update_right_requirements(self.required_columns, left_size);
            }
            _ => {}
        }
        self.required_columns
            .extend(collect_columns_in_join_conditions(
                smj.on(),
                None,
                left_size,
                self.children_nodes[0].plan.schema(),
                self.children_nodes[1].plan.schema(),
            ));
        let (analyzed_join_left, analyzed_join_right) = analyze_requirements_of_joins(
            smj.left(),
            smj.right(),
            &self.required_columns,
            left_size,
        );

        match smj.join_type() {
            JoinType::Inner | JoinType::Left | JoinType::Right | JoinType::Full => {
                match (
                    all_columns_required(&analyzed_join_left),
                    all_columns_required(&analyzed_join_right),
                ) {
                    // We need two projections on top of both children.
                    (false, false) => {
                        let new_on = update_equivalence_conditions(
                            smj.on(),
                            &analyzed_join_left,
                            &analyzed_join_right,
                        );
                        let new_filter = update_non_equivalence_conditions(
                            smj.filter.as_ref(),
                            &analyzed_join_left,
                            &analyzed_join_right,
                        );
                        let (new_left_child, new_right_child, schema_mapping) = self
                            .insert_multi_projections_below_join(
                                left_size,
                                analyzed_join_left,
                                analyzed_join_right,
                            )?;
                        let plan = Arc::new(SortMergeJoinExec::try_new(
                            new_left_child.plan.clone(),
                            new_right_child.plan.clone(),
                            new_on,
                            new_filter,
                            smj.join_type(),
                            smj.sort_options.clone(),
                            smj.null_equals_null,
                        )?) as _;

                        self = ProjectionOptimizer {
                            plan,
                            required_columns: HashSet::new(),
                            schema_mapping,
                            children_nodes: vec![new_left_child, new_right_child],
                        }
                    }
                    (false, true) => {
                        let required_columns = mem::take(&mut self.required_columns);
                        let mut right_child = self.children_nodes.swap_remove(1);
                        let new_on = update_equivalence_conditions(
                            smj.on(),
                            &analyzed_join_left,
                            &HashMap::new(),
                        );
                        let new_filter = update_non_equivalence_conditions(
                            smj.filter.as_ref(),
                            &analyzed_join_right,
                            &HashMap::new(),
                        );
                        let (new_left_child, mut left_schema_mapping) =
                            self.insert_projection_below_left_child(analyzed_join_left)?;
                        right_child.required_columns = required_columns
                            .iter()
                            .filter(|col| col.index() >= left_size)
                            .map(|col| Column::new(col.name(), col.index() - left_size))
                            .collect();
                        let plan = Arc::new(SortMergeJoinExec::try_new(
                            new_left_child.plan.clone(),
                            right_child.plan.clone(),
                            new_on,
                            new_filter,
                            smj.join_type(),
                            smj.sort_options.clone(),
                            smj.null_equals_null,
                        )?) as _;
                        let new_left_size = new_left_child.plan.schema().fields().len();
                        left_schema_mapping = extend_left_mapping_with_right(
                            left_schema_mapping,
                            &right_child.plan,
                            left_size,
                            new_left_size,
                        );
                        self = ProjectionOptimizer {
                            plan,
                            required_columns: HashSet::new(),
                            schema_mapping: left_schema_mapping,
                            children_nodes: vec![new_left_child, right_child],
                        }
                    }
                    (true, false) => {
                        let required_columns = mem::take(&mut self.required_columns);
                        let mut left_child = self.children_nodes.swap_remove(0);
                        let new_on = update_equivalence_conditions(
                            smj.on(),
                            &HashMap::new(),
                            &analyzed_join_right,
                        );
                        let new_filter = update_non_equivalence_conditions(
                            smj.filter.as_ref(),
                            &HashMap::new(),
                            &analyzed_join_right,
                        );
                        let (new_right_child, mut right_schema_mapping) = self
                            .insert_projection_below_right_child(analyzed_join_right)?;
                        right_schema_mapping =
                            update_right_mapping(right_schema_mapping, left_size);

                        left_child.required_columns =
                            collect_left_used_columns(required_columns, left_size);

                        let plan = Arc::new(SortMergeJoinExec::try_new(
                            left_child.plan.clone(),
                            new_right_child.plan.clone(),
                            new_on,
                            new_filter,
                            smj.join_type(),
                            smj.sort_options.clone(),
                            smj.null_equals_null,
                        )?) as _;

                        self = ProjectionOptimizer {
                            plan,
                            required_columns: HashSet::new(),
                            schema_mapping: right_schema_mapping,
                            children_nodes: vec![left_child, new_right_child],
                        }
                    }
                    // All columns are required.
                    (true, true) => {
                        self.required_columns = HashSet::new();
                        self.children_nodes.iter_mut().for_each(|c| {
                            c.required_columns = collect_columns_in_plan_schema(&c.plan);
                        })
                    }
                }
            }
            JoinType::LeftAnti | JoinType::LeftSemi => {
                match all_columns_required(&analyzed_join_left) {
                    false => {
                        let mut right_child = self.children_nodes.swap_remove(1);
                        let new_on = update_equivalence_conditions(
                            smj.on(),
                            &analyzed_join_left,
                            &HashMap::new(),
                        );
                        let new_filter = update_non_equivalence_conditions(
                            smj.filter.as_ref(),
                            &analyzed_join_left,
                            &HashMap::new(),
                        );
                        let (new_left_child, left_schema_mapping) =
                            self.insert_projection_below_left_child(analyzed_join_left)?;

                        let plan = Arc::new(SortMergeJoinExec::try_new(
                            new_left_child.plan.clone(),
                            right_child.plan.clone(),
                            new_on,
                            new_filter,
                            smj.join_type(),
                            smj.sort_options.clone(),
                            smj.null_equals_null,
                        )?) as _;

                        right_child.required_columns = analyzed_join_right
                            .into_iter()
                            .filter_map(
                                |(column, used)| if used { Some(column) } else { None },
                            )
                            .collect();
                        self = ProjectionOptimizer {
                            plan,
                            required_columns: HashSet::new(),
                            schema_mapping: left_schema_mapping,
                            children_nodes: vec![new_left_child, right_child],
                        }
                    }
                    true => {
                        self.children_nodes[0].required_columns =
                            collect_columns_in_plan_schema(&self.children_nodes[0].plan);
                        self.children_nodes[1].required_columns = analyzed_join_right
                            .into_iter()
                            .filter_map(
                                |(column, used)| if used { Some(column) } else { None },
                            )
                            .collect()
                    }
                }
            }
            JoinType::RightAnti | JoinType::RightSemi => {
                match all_columns_required(&analyzed_join_right) {
                    false => {
                        let mut left_child = self.children_nodes.swap_remove(0);
                        let new_on = update_equivalence_conditions(
                            smj.on(),
                            &HashMap::new(),
                            &analyzed_join_right,
                        );
                        let new_filter = update_non_equivalence_conditions(
                            smj.filter.as_ref(),
                            &HashMap::new(),
                            &analyzed_join_right,
                        );
                        let (new_right_child, right_schema_mapping) = self
                            .insert_projection_below_right_child(analyzed_join_right)?;

                        let plan = Arc::new(SortMergeJoinExec::try_new(
                            left_child.plan.clone(),
                            new_right_child.plan.clone(),
                            new_on,
                            new_filter,
                            smj.join_type(),
                            smj.sort_options.clone(),
                            smj.null_equals_null,
                        )?) as _;

                        left_child.required_columns = analyzed_join_left
                            .into_iter()
                            .filter_map(
                                |(column, used)| if used { Some(column) } else { None },
                            )
                            .collect();
                        self = ProjectionOptimizer {
                            plan,
                            required_columns: HashSet::new(),
                            schema_mapping: right_schema_mapping,
                            children_nodes: vec![left_child, new_right_child],
                        }
                    }
                    true => {
                        self.children_nodes[0].required_columns = analyzed_join_left
                            .into_iter()
                            .filter_map(
                                |(column, used)| if used { Some(column) } else { None },
                            )
                            .collect();
                        self.children_nodes[1].required_columns =
                            collect_columns_in_plan_schema(&self.children_nodes[1].plan);
                    }
                }
            }
        }
        Ok(self)
    }

    fn try_insert_below_symmetric_hash_join(
        mut self,
        shj: &SymmetricHashJoinExec,
    ) -> Result<ProjectionOptimizer> {
        let left_size = shj.left().schema().fields().len();
        // SymmetricHashJoinExec extends the requirements with the columns in its equivalence and non-equivalence conditions.
        match shj.join_type() {
            JoinType::RightAnti | JoinType::RightSemi => {
                self.required_columns =
                    update_right_requirements(self.required_columns, left_size);
            }
            _ => {}
        }
        self.required_columns
            .extend(collect_columns_in_join_conditions(
                shj.on(),
                shj.filter(),
                left_size,
                self.children_nodes[0].plan.schema(),
                self.children_nodes[1].plan.schema(),
            ));
        let (analyzed_join_left, analyzed_join_right) = analyze_requirements_of_joins(
            shj.left(),
            shj.right(),
            &self.required_columns,
            left_size,
        );

        match shj.join_type() {
            JoinType::Inner | JoinType::Left | JoinType::Right | JoinType::Full => {
                match (
                    all_columns_required(&analyzed_join_left),
                    all_columns_required(&analyzed_join_right),
                ) {
                    // We need two projections on top of both children.
                    (false, false) => {
                        let new_on = update_equivalence_conditions(
                            shj.on(),
                            &analyzed_join_left,
                            &analyzed_join_right,
                        );
                        let new_filter = update_non_equivalence_conditions(
                            shj.filter(),
                            &analyzed_join_left,
                            &analyzed_join_right,
                        );
                        let (new_left_child, new_right_child, schema_mapping) = self
                            .insert_multi_projections_below_join(
                                left_size,
                                analyzed_join_left,
                                analyzed_join_right,
                            )?;
                        let new_left_sort_exprs =
                            shj.left_sort_exprs().map(|sort_expr| {
                                update_sort_expressions(sort_expr, &schema_mapping)
                            });
                        let new_right_sort_exprs =
                            shj.left_sort_exprs().map(|sort_expr| {
                                update_sort_expressions(sort_expr, &schema_mapping)
                            });

                        let plan = Arc::new(SymmetricHashJoinExec::try_new(
                            new_left_child.plan.clone(),
                            new_right_child.plan.clone(),
                            new_on,
                            new_filter,
                            shj.join_type(),
                            shj.null_equals_null(),
                            new_left_sort_exprs,
                            new_right_sort_exprs,
                            shj.partition_mode(),
                        )?) as _;

                        self = ProjectionOptimizer {
                            plan,
                            required_columns: HashSet::new(),
                            schema_mapping,
                            children_nodes: vec![new_left_child, new_right_child],
                        }
                    }
                    (false, true) => {
                        let required_columns = mem::take(&mut self.required_columns);
                        let mut right_child = self.children_nodes.swap_remove(1);
                        let new_on = update_equivalence_conditions(
                            shj.on(),
                            &analyzed_join_left,
                            &HashMap::new(),
                        );
                        let new_filter = update_non_equivalence_conditions(
                            shj.filter(),
                            &analyzed_join_right,
                            &HashMap::new(),
                        );
                        let (new_left_child, mut left_schema_mapping) =
                            self.insert_projection_below_left_child(analyzed_join_left)?;

                        right_child.required_columns =
                            update_right_child_requirements(&required_columns, left_size);

                        let plan = Arc::new(SymmetricHashJoinExec::try_new(
                            new_left_child.plan.clone(),
                            right_child.plan.clone(),
                            new_on,
                            new_filter,
                            shj.join_type(),
                            shj.null_equals_null(),
                            shj.left_sort_exprs().map(|exprs| exprs.to_vec()),
                            shj.right_sort_exprs().map(|exprs| exprs.to_vec()),
                            shj.partition_mode(),
                        )?) as _;
                        let new_left_size = new_left_child.plan.schema().fields().len();
                        left_schema_mapping = extend_left_mapping_with_right(
                            left_schema_mapping,
                            &right_child.plan,
                            left_size,
                            new_left_size,
                        );
                        self = ProjectionOptimizer {
                            plan,
                            required_columns: HashSet::new(),
                            schema_mapping: left_schema_mapping,
                            children_nodes: vec![new_left_child, right_child],
                        }
                    }
                    (true, false) => {
                        let required_columns = mem::take(&mut self.required_columns);
                        let mut left_child = self.children_nodes.swap_remove(0);
                        let new_on = update_equivalence_conditions(
                            shj.on(),
                            &HashMap::new(),
                            &analyzed_join_right,
                        );
                        let new_filter = update_non_equivalence_conditions(
                            shj.filter(),
                            &HashMap::new(),
                            &analyzed_join_right,
                        );
                        let (new_right_child, mut right_schema_mapping) = self
                            .insert_projection_below_right_child(analyzed_join_right)?;

                        right_schema_mapping =
                            update_right_mapping(right_schema_mapping, left_size);

                        left_child.required_columns =
                            collect_left_used_columns(required_columns, left_size);
                        let plan = Arc::new(SymmetricHashJoinExec::try_new(
                            left_child.plan.clone(),
                            new_right_child.plan.clone(),
                            new_on,
                            new_filter,
                            shj.join_type(),
                            shj.null_equals_null(),
                            shj.left_sort_exprs().map(|exprs| exprs.to_vec()),
                            shj.right_sort_exprs().map(|exprs| exprs.to_vec()),
                            shj.partition_mode(),
                        )?) as _;

                        self = ProjectionOptimizer {
                            plan,
                            required_columns: HashSet::new(),
                            schema_mapping: right_schema_mapping,
                            children_nodes: vec![left_child, new_right_child],
                        }
                    }
                    // All columns are required.
                    (true, true) => {
                        self.required_columns = HashSet::new();
                        self.children_nodes.iter_mut().for_each(|c| {
                            c.required_columns = collect_columns_in_plan_schema(&c.plan);
                        })
                    }
                }
            }
            JoinType::LeftAnti | JoinType::LeftSemi => {
                match all_columns_required(&analyzed_join_left) {
                    false => {
                        let mut right_child = self.children_nodes.swap_remove(1);
                        let new_on = update_equivalence_conditions(
                            shj.on(),
                            &analyzed_join_left,
                            &HashMap::new(),
                        );
                        let new_filter = update_non_equivalence_conditions(
                            shj.filter(),
                            &analyzed_join_left,
                            &HashMap::new(),
                        );
                        let (new_left_child, left_schema_mapping) =
                            self.insert_projection_below_left_child(analyzed_join_left)?;

                        let plan = Arc::new(SymmetricHashJoinExec::try_new(
                            new_left_child.plan.clone(),
                            right_child.plan.clone(),
                            new_on,
                            new_filter,
                            shj.join_type(),
                            shj.null_equals_null(),
                            shj.left_sort_exprs().map(|exprs| exprs.to_vec()),
                            shj.right_sort_exprs().map(|exprs| exprs.to_vec()),
                            shj.partition_mode(),
                        )?) as _;

                        right_child.required_columns = analyzed_join_right
                            .into_iter()
                            .filter_map(
                                |(column, used)| if used { Some(column) } else { None },
                            )
                            .collect();
                        self = ProjectionOptimizer {
                            plan,
                            required_columns: HashSet::new(),
                            schema_mapping: left_schema_mapping,
                            children_nodes: vec![new_left_child, right_child],
                        }
                    }
                    true => {
                        self.children_nodes[0].required_columns =
                            collect_columns_in_plan_schema(&self.children_nodes[0].plan);
                        self.children_nodes[1].required_columns = analyzed_join_right
                            .into_iter()
                            .filter_map(
                                |(column, used)| if used { Some(column) } else { None },
                            )
                            .collect()
                    }
                }
            }
            JoinType::RightAnti | JoinType::RightSemi => {
                match all_columns_required(&analyzed_join_right) {
                    false => {
                        let mut left_child = self.children_nodes.swap_remove(0);
                        let new_on = update_equivalence_conditions(
                            shj.on(),
                            &HashMap::new(),
                            &analyzed_join_right,
                        );
                        let new_filter = update_non_equivalence_conditions(
                            shj.filter(),
                            &HashMap::new(),
                            &analyzed_join_right,
                        );
                        let (new_right_child, right_schema_mapping) = self
                            .insert_projection_below_right_child(analyzed_join_right)?;

                        let plan = Arc::new(SymmetricHashJoinExec::try_new(
                            left_child.plan.clone(),
                            new_right_child.plan.clone(),
                            new_on,
                            new_filter,
                            shj.join_type(),
                            shj.null_equals_null(),
                            shj.left_sort_exprs().map(|exprs| exprs.to_vec()),
                            shj.right_sort_exprs().map(|exprs| exprs.to_vec()),
                            shj.partition_mode(),
                        )?) as _;

                        left_child.required_columns = analyzed_join_left
                            .into_iter()
                            .filter_map(
                                |(column, used)| if used { Some(column) } else { None },
                            )
                            .collect();
                        self = ProjectionOptimizer {
                            plan,
                            required_columns: HashSet::new(),
                            schema_mapping: right_schema_mapping,
                            children_nodes: vec![left_child, new_right_child],
                        }
                    }
                    true => {
                        self.children_nodes[0].required_columns = analyzed_join_left
                            .into_iter()
                            .filter_map(
                                |(column, used)| if used { Some(column) } else { None },
                            )
                            .collect();
                        self.children_nodes[1].required_columns =
                            collect_columns_in_plan_schema(&self.children_nodes[1].plan);
                    }
                }
            }
        }
        Ok(self)
    }

    fn try_insert_below_aggregate(
        mut self,
        agg: &AggregateExec,
    ) -> Result<ProjectionOptimizer> {
        // `AggregateExec` applies their own projections. We can only limit
        // the aggregate expressions unless they are used in the upper plans.
        let group_columns_len = agg.group_expr().expr().len();
        let required_indices = self
            .required_columns
            .iter()
            .map(|req_col| req_col.index())
            .collect::<HashSet<_>>();
        let unused_aggr_exprs = agg
            .aggr_expr()
            .iter()
            .enumerate()
            .filter(|(idx, _expr)| !required_indices.contains(&(idx + group_columns_len)))
            .map(|(idx, _expr)| idx)
            .collect::<HashSet<usize>>();

        if !unused_aggr_exprs.is_empty() {
            let new_plan = AggregateExec::try_new(
                *agg.mode(),
                agg.group_expr().clone(),
                agg.aggr_expr()
                    .iter()
                    .enumerate()
                    .filter(|(idx, _expr)| !unused_aggr_exprs.contains(idx))
                    .map(|(_idx, expr)| expr.clone())
                    .collect(),
                agg.filter_expr().to_vec(),
                agg.input().clone(),
                agg.input_schema(),
            )?;
            self.children_nodes[0].required_columns = new_plan
                .group_expr()
                .expr()
                .iter()
                .flat_map(|(e, _alias)| collect_columns(e))
                .collect();
            self.children_nodes[0].required_columns.extend(
                new_plan.aggr_expr().iter().flat_map(|e| {
                    e.expressions()
                        .iter()
                        .flat_map(collect_columns)
                        .collect::<HashSet<Column>>()
                }),
            );
            self.plan = Arc::new(new_plan);
            self.required_columns = HashSet::new();
        } else {
            match agg.mode() {
                AggregateMode::Final | AggregateMode::FinalPartitioned => {
                    let mut group_expr_len = agg.group_expr().expr().iter().count();
                    let aggr_columns = agg
                        .aggr_expr()
                        .iter()
                        .flat_map(|e| {
                            e.state_fields().map(|field| {
                                field
                                    .iter()
                                    .map(|field| {
                                        group_expr_len += 1;
                                        Column::new(field.name(), group_expr_len - 1)
                                    })
                                    .collect::<Vec<_>>()
                            })
                        })
                        .flatten()
                        .collect::<Vec<_>>();
                    let group_columns = agg
                        .group_expr()
                        .expr()
                        .iter()
                        .flat_map(|(expr, _name)| collect_columns(expr))
                        .collect::<Vec<_>>();
                    let filter_columns = agg
                        .filter_expr()
                        .iter()
                        .filter_map(|expr| expr.as_ref().map(collect_columns))
                        .flatten()
                        .collect::<Vec<_>>();
                    self.children_nodes[0].required_columns.extend(
                        aggr_columns
                            .into_iter()
                            .chain(group_columns)
                            .chain(filter_columns),
                    )
                }
                _ => {
                    let aggr_columns = agg
                        .aggr_expr()
                        .iter()
                        .flat_map(|e| {
                            e.expressions()
                                .iter()
                                .flat_map(collect_columns)
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>();
                    let group_columns = agg
                        .group_expr()
                        .expr()
                        .iter()
                        .flat_map(|(expr, _name)| collect_columns(expr))
                        .collect::<Vec<_>>();
                    let filter_columns = agg
                        .filter_expr()
                        .iter()
                        .filter_map(|expr| expr.as_ref().map(collect_columns))
                        .flatten()
                        .collect::<Vec<_>>();
                    self.children_nodes[0].required_columns.extend(
                        aggr_columns
                            .into_iter()
                            .chain(group_columns)
                            .chain(filter_columns),
                    );
                }
            };
        }
        Ok(self)
    }

    fn try_insert_below_window_aggregate(
        mut self,
        w_agg: &WindowAggExec,
    ) -> Result<ProjectionOptimizer> {
        // Both tries to insert a projection to narrow input columns, and tries to narrow the window
        // expressions. If none of them survives, we can even remove the window execution plan.
        self.required_columns
            .extend(w_agg.window_expr().iter().flat_map(|window_expr| {
                window_expr
                    .expressions()
                    .iter()
                    .flat_map(collect_columns)
                    .collect::<Vec<_>>()
            }));
        self.required_columns
            .extend(w_agg.partition_keys.iter().flat_map(collect_columns));
        let requirement_map = analyze_requirements(&self);
        if !all_columns_required(&requirement_map) {
            if window_agg_required(
                w_agg.input().schema().fields().len(),
                &requirement_map,
            ) {
                if w_agg
                    .window_expr()
                    .iter()
                    .any(|expr| expr.clone().with_new_expressions(vec![]).is_none())
                {
                    self.children_nodes[0].required_columns = self
                        .required_columns
                        .iter()
                        .filter(|col| {
                            col.index()
                                < w_agg.schema().fields().len()
                                    - w_agg.window_expr().len()
                        })
                        .cloned()
                        .collect();
                    return Ok(self);
                }
                let (new_child, schema_mapping, window_usage) =
                    self.insert_projection_below_window(w_agg, requirement_map)?;
                // Rewrite the sort expressions with possibly updated column indices.
                let Some(new_window_exprs) = w_agg
                    .window_expr()
                    .iter()
                    .zip(window_usage.clone())
                    .filter(|(_window_expr, (_window_col, usage))| *usage)
                    .map(|(window_expr, (_window_col, _usage))| {
                        let new_exprs = update_expressions(
                            &window_expr.expressions(),
                            &HashMap::new(),
                        );
                        window_expr.clone().with_new_expressions(new_exprs)
                    })
                    .collect::<Option<Vec<_>>>()
                else {
                    return Err(datafusion_common::DataFusionError::Internal(
                        "".to_string(),
                    ));
                };

                let new_keys = w_agg
                    .partition_keys
                    .iter()
                    .zip(window_usage)
                    .filter_map(|(key, (_column, usage))| {
                        if usage {
                            Some(update_column_index(key, &schema_mapping))
                        } else {
                            None
                        }
                    })
                    .collect();
                let plan = Arc::new(WindowAggExec::try_new(
                    new_window_exprs,
                    new_child.plan.clone(),
                    new_keys,
                )?) as _;
                let required_columns = collect_columns_in_plan_schema(&plan);
                self = ProjectionOptimizer {
                    plan,
                    required_columns,
                    schema_mapping,
                    children_nodes: vec![new_child],
                }
            } else {
                // Remove the WindowAggExec
                self = self.children_nodes.swap_remove(0);
                self.required_columns = requirement_map
                    .into_iter()
                    .filter_map(|(column, used)| if used { Some(column) } else { None })
                    .collect();
            }
        } else {
            self.children_nodes[0].required_columns =
                mem::take(&mut self.required_columns)
                    .into_iter()
                    .filter(|col| {
                        col.index()
                            < w_agg.schema().fields().len() - w_agg.window_expr().len()
                    })
                    .collect();
        }
        Ok(self)
    }

    fn try_insert_below_bounded_window_aggregate(
        mut self,
        bw_agg: &BoundedWindowAggExec,
    ) -> Result<ProjectionOptimizer> {
        // Both tries to insert a projection to narrow input columns, and tries to narrow the window
        // expressions. If none of them survives, we can even remove the window execution plan.
        self.required_columns
            .extend(bw_agg.window_expr().iter().flat_map(|window_expr| {
                window_expr
                    .expressions()
                    .iter()
                    .flat_map(collect_columns)
                    .collect::<Vec<_>>()
            }));
        self.required_columns
            .extend(bw_agg.partition_keys.iter().flat_map(collect_columns));
        let requirement_map = analyze_requirements(&self);
        if !all_columns_required(&requirement_map) {
            if window_agg_required(
                bw_agg.input().schema().fields().len(),
                &requirement_map,
            ) {
                if bw_agg
                    .window_expr()
                    .iter()
                    .any(|expr| expr.clone().with_new_expressions(vec![]).is_none())
                {
                    self.children_nodes[0].required_columns =
                        mem::take(&mut self.required_columns)
                            .into_iter()
                            .filter(|col| {
                                col.index()
                                    < bw_agg.schema().fields().len()
                                        - bw_agg.window_expr().len()
                            })
                            .collect();
                    return Ok(self);
                }
                let (new_child, schema_mapping, window_usage) =
                    self.insert_projection_below_bounded_window(bw_agg, requirement_map)?;
                // Rewrite the sort expressions with possibly updated column indices.
                let Some(new_window_exprs) = bw_agg
                    .window_expr()
                    .iter()
                    .zip(window_usage.clone())
                    .filter(|(_window_expr, (_window_col, usage))| *usage)
                    .map(|(window_expr, (_window_col, _usage))| {
                        let new_exprs = update_expressions(
                            &window_expr.expressions(),
                            &schema_mapping,
                        );
                        window_expr.clone().with_new_expressions(new_exprs)
                    })
                    .collect::<Option<Vec<_>>>()
                else {
                    return Err(datafusion_common::DataFusionError::Internal(
                        "".to_string(),
                    ));
                };

                let new_keys = bw_agg
                    .partition_keys
                    .iter()
                    .zip(window_usage)
                    .filter_map(|(key, (_column, usage))| {
                        if usage {
                            Some(update_column_index(key, &schema_mapping))
                        } else {
                            None
                        }
                    })
                    .collect();
                let plan = Arc::new(BoundedWindowAggExec::try_new(
                    new_window_exprs,
                    new_child.plan.clone(),
                    new_keys,
                    bw_agg.input_order_mode.clone(),
                )?) as _;
                let required_columns = collect_columns_in_plan_schema(&plan);
                self = ProjectionOptimizer {
                    plan,
                    required_columns,
                    schema_mapping,
                    children_nodes: vec![new_child],
                }
            } else {
                // Remove the WindowAggExec
                self = self.children_nodes.swap_remove(0);
                self.required_columns = requirement_map
                    .into_iter()
                    .filter_map(|(column, used)| if used { Some(column) } else { None })
                    .collect();
            }
        } else {
            self.children_nodes[0].required_columns =
                mem::take(&mut self.required_columns)
                    .into_iter()
                    .filter(|col| {
                        col.index()
                            < bw_agg.schema().fields().len() - bw_agg.window_expr().len()
                    })
                    .collect();
        }
        Ok(self)
    }

    /// If a node is known to have redundant columns, we need to insert a projection to its input.
    /// This function takes this node and requirement mapping of this node. Then, defines the projection
    /// and constructs the new subtree. The returned objects are the new tree starting from the inserted
    /// projection, and the mapping of columns referring to the schemas of pre-insertion and post-insertion.
    fn insert_projection(
        self,
        requirement_map: ColumnRequirements,
    ) -> Result<(Self, HashMap<Column, Column>)> {
        // During the iteration, we construct the ProjectionExec with required columns as the new child,
        // and also collect the unused columns to store the index changes after removal of some columns.
        let (used_columns, unused_columns) = split_column_requirements(&requirement_map);
        let projected_exprs = convert_projection_exprs(used_columns);
        let inserted_projection = Arc::new(ProjectionExec::try_new(
            projected_exprs,
            self.plan.children()[0].clone(),
        )?) as _;

        let new_mapping =
            calculate_column_mapping(&self.required_columns, &unused_columns);

        let new_requirements = collect_columns_in_plan_schema(&inserted_projection);
        let inserted_projection = ProjectionOptimizer {
            plan: inserted_projection,
            // Required columns must have been extended with self node requirements before this point.
            required_columns: new_requirements,
            schema_mapping: HashMap::new(),
            children_nodes: self.children_nodes,
        };
        Ok((inserted_projection, new_mapping))
    }

    /// Multi-child version of `insert_projection` for `UnionExec`'s.
    fn insert_multi_projection_below_union(
        self,
        requirement_map: ColumnRequirements,
    ) -> Result<(Vec<Self>, HashMap<Column, Column>)> {
        // During the iteration, we construct the ProjectionExec's with required columns as the new children,
        // and also collect the unused columns to store the index changes after removal of some columns.
        let (used_columns, unused_columns) = split_column_requirements(&requirement_map);
        let projected_exprs = convert_projection_exprs(used_columns);

        let inserted_projections = self
            .plan
            .children()
            .into_iter()
            .map(|child_plan| {
                Ok(Arc::new(ProjectionExec::try_new(
                    projected_exprs.clone(),
                    child_plan,
                )?) as _)
            })
            .collect::<Result<Vec<_>>>()?;

        let new_mapping =
            calculate_column_mapping(&self.required_columns, &unused_columns);

        let new_requirements = inserted_projections
            .iter()
            .map(|inserted_projection| {
                collect_columns_in_plan_schema(inserted_projection)
            })
            .collect::<Vec<_>>();
        let inserted_projection_nodes = inserted_projections
            .into_iter()
            .zip(self.children_nodes)
            .enumerate()
            .map(|(idx, (p, child))| ProjectionOptimizer {
                plan: p,
                required_columns: new_requirements[idx].clone(),
                schema_mapping: HashMap::new(),
                children_nodes: vec![child],
            })
            .collect();
        Ok((inserted_projection_nodes, new_mapping))
    }

    /// Single child version of `insert_projection` for joins.
    #[allow(clippy::type_complexity)]
    fn insert_projection_below_multi_child(
        mut self,
        requirement_map_left: ColumnRequirements,
        requirement_map_right: ColumnRequirements,
    ) -> Result<(Self, Self, HashMap<Column, Column>, HashMap<Column, Column>)> {
        // During the iteration, we construct the ProjectionExec with required columns as the new child,
        // and also collect the unused columns to store the index changes after removal of some columns.
        let (used_columns, unused_columns) =
            split_column_requirements(&requirement_map_left);
        let child_plan = self.plan.children().remove(0);
        let new_left_mapping = calculate_column_mapping(&used_columns, &unused_columns);
        let projected_exprs = convert_projection_exprs(used_columns);
        let inserted_projection =
            Arc::new(ProjectionExec::try_new(projected_exprs, child_plan)?) as _;

        let required_columns = collect_columns_in_plan_schema(&inserted_projection);
        let left_inserted_projection = ProjectionOptimizer {
            plan: inserted_projection,
            required_columns,
            schema_mapping: HashMap::new(),
            children_nodes: vec![self.children_nodes.swap_remove(0)],
        };

        let (used_columns, unused_columns) =
            split_column_requirements(&requirement_map_right);
        let child_plan = self.plan.children().remove(1);
        let new_right_mapping = calculate_column_mapping(&used_columns, &unused_columns);
        let projected_exprs = convert_projection_exprs(used_columns);
        let inserted_projection =
            Arc::new(ProjectionExec::try_new(projected_exprs, child_plan)?) as _;

        let required_columns = collect_columns_in_plan_schema(&inserted_projection);
        let right_inserted_projection = ProjectionOptimizer {
            plan: inserted_projection,
            required_columns,
            schema_mapping: HashMap::new(),
            children_nodes: vec![self.children_nodes.swap_remove(0)],
        };
        Ok((
            left_inserted_projection,
            right_inserted_projection,
            new_left_mapping,
            new_right_mapping,
        ))
    }

    /// Left child version of `insert_projection` for joins.
    fn insert_projection_below_left_child(
        mut self,
        requirement_map_left: ColumnRequirements,
    ) -> Result<(Self, HashMap<Column, Column>)> {
        // During the iteration, we construct the ProjectionExec with required columns as the new child,
        // and also collect the unused columns to store the index changes after removal of some columns.
        let (used_columns, unused_columns) =
            split_column_requirements(&requirement_map_left);
        let child_plan = self.plan.children().remove(0);
        let new_mapping = calculate_column_mapping(&used_columns, &unused_columns);
        let projected_exprs = convert_projection_exprs(used_columns);
        let inserted_projection =
            Arc::new(ProjectionExec::try_new(projected_exprs, child_plan)?) as _;

        let required_columns = collect_columns_in_plan_schema(&inserted_projection);
        let inserted_projection = ProjectionOptimizer {
            plan: inserted_projection,
            required_columns,
            schema_mapping: HashMap::new(),
            children_nodes: vec![self.children_nodes.swap_remove(0)],
        };
        Ok((inserted_projection, new_mapping))
    }

    /// Right child version of `insert_projection` for joins.
    fn insert_projection_below_right_child(
        mut self,
        requirement_map_right: ColumnRequirements,
    ) -> Result<(Self, HashMap<Column, Column>)> {
        // During the iteration, we construct the ProjectionExec with required columns as the new child,
        // and also collect the unused columns to store the index changes after removal of some columns.
        let (used_columns, unused_columns) =
            split_column_requirements(&requirement_map_right);
        let child_plan = self.plan.children().remove(1);
        let new_mapping = calculate_column_mapping(&used_columns, &unused_columns);
        let projected_exprs = convert_projection_exprs(used_columns);
        let inserted_projection =
            Arc::new(ProjectionExec::try_new(projected_exprs, child_plan)?) as _;

        let required_columns = collect_columns_in_plan_schema(&inserted_projection);
        let inserted_projection = ProjectionOptimizer {
            plan: inserted_projection,
            required_columns,
            schema_mapping: HashMap::new(),
            children_nodes: vec![self.children_nodes.swap_remove(0)],
        };
        Ok((inserted_projection, new_mapping))
    }

    /// Multi-child version of `insert_projection` for joins.
    fn insert_multi_projections_below_join(
        self,
        left_size: usize,
        requirement_map_left: ColumnRequirements,
        requirement_map_right: ColumnRequirements,
    ) -> Result<(Self, Self, HashMap<Column, Column>)> {
        let original_right = self.children_nodes[1].plan.clone();
        let (
            new_left_child,
            new_right_child,
            mut left_schema_mapping,
            right_schema_mapping,
        ) = self.insert_projection_below_multi_child(
            requirement_map_left,
            requirement_map_right,
        )?;

        let new_left_size = new_left_child.plan.schema().fields().len();
        // left_schema_mapping does not need to be change, but it is updated with
        // those coming form the right side to represent overall join output mapping.
        for (idx, field) in
            original_right
                .schema()
                .fields()
                .iter()
                .enumerate()
                .filter(|(idx, field)| {
                    let right_projection =
                        collect_columns_in_plan_schema(&new_right_child.plan);
                    right_projection.contains(&Column::new(field.name(), *idx))
                })
        {
            left_schema_mapping.insert(
                Column::new(field.name(), idx + left_size),
                Column::new(field.name(), idx + new_left_size),
            );
        }
        for (old, new) in right_schema_mapping.into_iter() {
            left_schema_mapping.insert(
                Column::new(old.name(), old.index() + left_size),
                Column::new(new.name(), new.index() + new_left_size),
            );
        }
        Ok((new_left_child, new_right_child, left_schema_mapping))
    }

    /// `insert_projection` for windows.
    fn insert_projection_below_window(
        self,
        w_agg: &WindowAggExec,
        requirement_map: ColumnRequirements,
    ) -> Result<(Self, HashMap<Column, Column>, ColumnRequirements)> {
        let original_schema_len = w_agg.schema().fields().len();
        let (base, window): (ColumnRequirements, ColumnRequirements) = requirement_map
            .into_iter()
            .partition(|(column, _used)| column.index() < original_schema_len);
        let (used_columns, mut unused_columns) = split_column_requirements(&base);
        let projected_exprs = convert_projection_exprs(used_columns);

        window.iter().for_each(|(col, used)| {
            if !used {
                unused_columns.insert(col.clone());
            }
        });
        let inserted_projection = Arc::new(ProjectionExec::try_new(
            projected_exprs,
            self.plan.children()[0].clone(),
        )?) as _;

        let new_mapping =
            calculate_column_mapping(&self.required_columns, &unused_columns);

        let new_requirements = collect_columns_in_plan_schema(&inserted_projection);
        let inserted_projection = ProjectionOptimizer {
            plan: inserted_projection,
            // Required columns must have been extended with self node requirements before this point.
            required_columns: new_requirements,
            schema_mapping: HashMap::new(),
            children_nodes: self.children_nodes,
        };
        Ok((inserted_projection, new_mapping, window))
    }

    /// `insert_projection` for bounded windows.
    fn insert_projection_below_bounded_window(
        self,
        bw_agg: &BoundedWindowAggExec,
        requirement_map: ColumnRequirements,
    ) -> Result<(Self, HashMap<Column, Column>, ColumnRequirements)> {
        let original_schema_len = bw_agg.schema().fields().len();
        let (base, window): (ColumnRequirements, ColumnRequirements) = requirement_map
            .into_iter()
            .partition(|(column, _used)| column.index() < original_schema_len);
        // let mut unused_columns = HashSet::new();
        let (required_cols, mut unused_columns) = split_column_requirements(&base);
        let projected_exprs = convert_projection_exprs(required_cols);
        window.iter().for_each(|(col, used)| {
            if !used {
                unused_columns.insert(col.clone());
            }
        });
        let inserted_projection = Arc::new(ProjectionExec::try_new(
            projected_exprs,
            self.plan.children()[0].clone(),
        )?) as _;

        let new_mapping =
            calculate_column_mapping(&self.required_columns, &unused_columns);

        let new_requirements = collect_columns_in_plan_schema(&inserted_projection);
        let inserted_projection = ProjectionOptimizer {
            plan: inserted_projection,
            // Required columns must have been extended with self node requirements before this point.
            required_columns: new_requirements,
            schema_mapping: HashMap::new(),
            children_nodes: self.children_nodes,
        };
        Ok((inserted_projection, new_mapping, window))
    }

    /// Responsible for updating the node's plan with new children and possibly updated column indices,
    /// and for transferring the column mapping to the upper nodes. There is an exception for the
    /// projection nodes; they may be removed also in case of being considered as unnecessary,
    /// which leads to re-update the mapping after removal.
    fn index_updater(mut self: ProjectionOptimizer) -> Result<Transformed<Self>> {
        let mut all_mappings = self
            .children_nodes
            .iter_mut()
            .map(|node| mem::take(&mut node.schema_mapping))
            .collect::<Vec<_>>();
        if !all_mappings.iter().all(|map| map.is_empty()) {
            // The self plan will update its column indices according to the changes its children schemas.
            let plan_copy = self.plan.clone();
            let plan_any = plan_copy.as_any();

            // These plans do not have any expression related field.
            // They simply transfer the mapping to the parent node.
            if let Some(_coal_batches) = plan_any.downcast_ref::<CoalesceBatchesExec>() {
                self.plan = self.plan.with_new_children(
                    self.children_nodes
                        .iter()
                        .map(|child| child.plan.clone())
                        .collect(),
                )?;
                update_mapping(&mut self, all_mappings)
            } else if let Some(_coal_parts) =
                plan_any.downcast_ref::<CoalescePartitionsExec>()
            {
                self.plan = self.plan.with_new_children(
                    self.children_nodes
                        .iter()
                        .map(|child| child.plan.clone())
                        .collect(),
                )?;
                update_mapping(&mut self, all_mappings)
            } else if let Some(_glimit) = plan_any.downcast_ref::<GlobalLimitExec>() {
                self.plan = self.plan.with_new_children(
                    self.children_nodes
                        .iter()
                        .map(|child| child.plan.clone())
                        .collect(),
                )?;
                update_mapping(&mut self, all_mappings)
            } else if let Some(_llimit) = plan_any.downcast_ref::<LocalLimitExec>() {
                self.plan = self.plan.with_new_children(
                    self.children_nodes
                        .iter()
                        .map(|child| child.plan.clone())
                        .collect(),
                )?;
                update_mapping(&mut self, all_mappings)
            } else if let Some(_union) = plan_any.downcast_ref::<UnionExec>() {
                self.plan = self.plan.with_new_children(
                    self.children_nodes
                        .iter()
                        .map(|child| child.plan.clone())
                        .collect(),
                )?;
                update_mapping(&mut self, all_mappings)
            } else if let Some(_union) = plan_any.downcast_ref::<InterleaveExec>() {
                self.plan = self.plan.with_new_children(
                    self.children_nodes
                        .iter()
                        .map(|child| child.plan.clone())
                        .collect(),
                )?;
                update_mapping(&mut self, all_mappings)
            } else if let Some(_cj) = plan_any.downcast_ref::<CrossJoinExec>() {
                self.plan = self.plan.with_new_children(
                    self.children_nodes
                        .iter()
                        .map(|child| child.plan.clone())
                        .collect(),
                )?;
                update_mapping(&mut self, all_mappings)
            } else if let Some(projection) = plan_any.downcast_ref::<ProjectionExec>() {
                self.plan = rewrite_projection(
                    projection,
                    self.children_nodes[0].plan.clone(),
                    &all_mappings[0],
                )?;

                // Rewriting the projection does not change its output schema,
                // and projections does not need to transfer the mapping to upper nodes.
            } else if let Some(filter) = plan_any.downcast_ref::<FilterExec>() {
                self.plan = rewrite_filter(
                    filter.predicate(),
                    self.children_nodes[0].plan.clone(),
                    &all_mappings[0],
                )?;
                update_mapping(&mut self, all_mappings)
            } else if let Some(repartition) = plan_any.downcast_ref::<RepartitionExec>() {
                self.plan = rewrite_repartition(
                    repartition.partitioning(),
                    self.children_nodes[0].plan.clone(),
                    &all_mappings[0],
                )?;
                update_mapping(&mut self, all_mappings)
            } else if let Some(sort) = plan_any.downcast_ref::<SortExec>() {
                self.plan = rewrite_sort(
                    sort,
                    self.children_nodes[0].plan.clone(),
                    &all_mappings[0],
                )?;
                update_mapping(&mut self, all_mappings)
            } else if let Some(sortp_merge) =
                plan_any.downcast_ref::<SortPreservingMergeExec>()
            {
                self.plan = rewrite_sort_preserving_merge(
                    sortp_merge,
                    self.children_nodes[0].plan.clone(),
                    &all_mappings[0],
                )?;
                update_mapping(&mut self, all_mappings)
            } else if let Some(hj) = plan_any.downcast_ref::<HashJoinExec>() {
                let left_mapping = all_mappings.swap_remove(0);
                let right_mapping = all_mappings.swap_remove(0);
                let projection = hj
                    .projection
                    .clone()
                    .unwrap_or((0..hj.schema().fields().len()).collect());
                let new_on = update_join_on(hj.on(), &left_mapping, &right_mapping);
                let new_filter = hj.filter().map(|filter| {
                    JoinFilter::new(
                        filter.expression().clone(),
                        filter
                            .column_indices()
                            .iter()
                            .map(|col_idx| match col_idx.side {
                                JoinSide::Left => ColumnIndex {
                                    index: left_mapping
                                        .iter()
                                        .find(|(old_column, _new_column)| {
                                            old_column.index() == col_idx.index
                                        })
                                        .map(|(_old_column, new_column)| {
                                            new_column.index()
                                        })
                                        .unwrap_or(col_idx.index),
                                    side: JoinSide::Left,
                                },
                                JoinSide::Right => ColumnIndex {
                                    index: right_mapping
                                        .iter()
                                        .find(|(old_column, _new_column)| {
                                            old_column.index() == col_idx.index
                                        })
                                        .map(|(_old_column, new_column)| {
                                            new_column.index()
                                        })
                                        .unwrap_or(col_idx.index),
                                    side: JoinSide::Right,
                                },
                            })
                            .collect(),
                        filter.schema().clone(),
                    )
                });

                match hj.join_type() {
                    JoinType::Inner
                    | JoinType::Left
                    | JoinType::Right
                    | JoinType::Full => {
                        let index_mapping = left_mapping
                            .iter()
                            .map(|(col1, col2)| (col1.index(), col2.index()))
                            .chain(right_mapping.iter().map(|(col1, col2)| {
                                (
                                    col1.index()
                                        + hj.children()[0].schema().fields().len(),
                                    col2.index()
                                        + self.children_nodes[0]
                                            .plan
                                            .schema()
                                            .fields()
                                            .len(),
                                )
                            }))
                            .collect::<HashMap<_, _>>();
                        let new_projection = projection
                            .into_iter()
                            .map(|idx| *index_mapping.get(&idx).unwrap_or(&idx))
                            .collect::<Vec<_>>();
                        let some_projection = new_projection
                            .first()
                            .map(|first| *first != 0)
                            .unwrap_or(true)
                            || !new_projection.windows(2).all(|w| w[0] + 1 == w[1]);

                        self.plan = HashJoinExec::try_new(
                            self.children_nodes[0].plan.clone(),
                            self.children_nodes[1].plan.clone(),
                            new_on,
                            new_filter,
                            hj.join_type(),
                            if some_projection {
                                Some(new_projection)
                            } else {
                                None
                            },
                            *hj.partition_mode(),
                            hj.null_equals_null(),
                        )
                        .map(|plan| Arc::new(plan) as _)?;
                        self.schema_mapping = left_mapping
                            .into_iter()
                            .chain(right_mapping.iter().map(|(col1, col2)| {
                                (
                                    Column::new(
                                        col1.name(),
                                        col1.index()
                                            + hj.children()[0].schema().fields().len(),
                                    ),
                                    Column::new(
                                        col2.name(),
                                        col2.index()
                                            + self.children_nodes[0]
                                                .plan
                                                .schema()
                                                .fields()
                                                .len(),
                                    ),
                                )
                            }))
                            .collect::<HashMap<_, _>>();
                    }
                    JoinType::LeftSemi | JoinType::LeftAnti => {
                        let index_mapping = left_mapping
                            .iter()
                            .map(|(col1, col2)| (col1.index(), col2.index()))
                            .collect::<HashMap<_, _>>();
                        let new_projection = projection
                            .into_iter()
                            .map(|idx| *index_mapping.get(&idx).unwrap_or(&idx))
                            .collect::<Vec<_>>();
                        let some_projection = new_projection
                            .first()
                            .map(|first| *first != 0)
                            .unwrap_or(true)
                            || !new_projection.windows(2).all(|w| w[0] + 1 == w[1]);

                        self.plan = HashJoinExec::try_new(
                            self.children_nodes[0].plan.clone(),
                            self.children_nodes[1].plan.clone(),
                            new_on,
                            new_filter,
                            hj.join_type(),
                            if some_projection {
                                Some(new_projection)
                            } else {
                                None
                            },
                            *hj.partition_mode(),
                            hj.null_equals_null(),
                        )
                        .map(|plan| Arc::new(plan) as _)?;
                        self.schema_mapping = left_mapping;
                    }
                    JoinType::RightSemi | JoinType::RightAnti => {
                        let index_mapping = right_mapping
                            .iter()
                            .map(|(col1, col2)| (col1.index(), col2.index()))
                            .collect::<HashMap<_, _>>();

                        let mut new_projection = projection
                            .into_iter()
                            .map(|idx| *index_mapping.get(&idx).unwrap_or(&idx))
                            .collect::<Vec<_>>();
                        new_projection.sort_by_key(|ind| *ind);
                        let some_projection = new_projection
                            .first()
                            .map(|first| *first != 0)
                            .unwrap_or(true)
                            || !new_projection.windows(2).all(|w| w[0] + 1 == w[1]);

                        self.plan = HashJoinExec::try_new(
                            self.children_nodes[0].plan.clone(),
                            self.children_nodes[1].plan.clone(),
                            new_on,
                            new_filter,
                            hj.join_type(),
                            if some_projection {
                                Some(new_projection)
                            } else {
                                None
                            },
                            *hj.partition_mode(),
                            hj.null_equals_null(),
                        )
                        .map(|plan| Arc::new(plan) as _)?;
                        self.schema_mapping = right_mapping;
                    }
                }
            } else if let Some(nlj) = plan_any.downcast_ref::<NestedLoopJoinExec>() {
                let left_size = self.children_nodes[0].plan.schema().fields().len();
                let left_mapping = all_mappings.swap_remove(0);
                let right_mapping = all_mappings.swap_remove(0);
                let new_mapping = left_mapping
                    .iter()
                    .map(|(initial, new)| (initial.clone(), new.clone()))
                    .chain(right_mapping.iter().map(|(initial, new)| {
                        (
                            Column::new(initial.name(), initial.index() + left_size),
                            Column::new(new.name(), new.index() + left_size),
                        )
                    }))
                    .collect::<HashMap<_, _>>();
                self.plan = rewrite_nested_loop_join(
                    nlj,
                    self.children_nodes[0].plan.clone(),
                    self.children_nodes[1].plan.clone(),
                    &left_mapping,
                    &right_mapping,
                    left_size,
                )?;
                match nlj.join_type() {
                    JoinType::Right
                    | JoinType::Full
                    | JoinType::Left
                    | JoinType::Inner => {
                        let (new_left, new_right) =
                            new_mapping.into_iter().partition(|(col_initial, _)| {
                                col_initial.index() < left_size
                            });
                        all_mappings.push(new_left);
                        all_mappings[0].extend(new_right);
                    }
                    JoinType::LeftSemi | JoinType::LeftAnti => {
                        all_mappings.push(left_mapping)
                    }
                    JoinType::RightAnti | JoinType::RightSemi => {
                        all_mappings.push(right_mapping)
                    }
                };
                update_mapping(&mut self, all_mappings)
            } else if let Some(smj) = plan_any.downcast_ref::<SortMergeJoinExec>() {
                let left_size = self.children_nodes[0].plan.schema().fields().len();
                let left_mapping = all_mappings.swap_remove(0);
                let right_mapping = all_mappings.swap_remove(0);
                let new_mapping = left_mapping
                    .iter()
                    .map(|(initial, new)| (initial.clone(), new.clone()))
                    .chain(right_mapping.iter().map(|(initial, new)| {
                        (
                            Column::new(initial.name(), initial.index() + left_size),
                            Column::new(new.name(), new.index() + left_size),
                        )
                    }))
                    .collect::<HashMap<_, _>>();
                self.plan = rewrite_sort_merge_join(
                    smj,
                    self.children_nodes[0].plan.clone(),
                    self.children_nodes[1].plan.clone(),
                    &left_mapping,
                    &right_mapping,
                    left_size,
                )?;
                match smj.join_type() {
                    JoinType::Right
                    | JoinType::Full
                    | JoinType::Left
                    | JoinType::Inner => {
                        let (new_left, new_right) =
                            new_mapping.into_iter().partition(|(col_initial, _)| {
                                col_initial.index() < left_size
                            });
                        all_mappings.push(new_left);
                        all_mappings[0].extend(new_right);
                    }
                    JoinType::LeftSemi | JoinType::LeftAnti => {
                        all_mappings.push(left_mapping)
                    }
                    JoinType::RightAnti | JoinType::RightSemi => {
                        all_mappings.push(right_mapping)
                    }
                };
                update_mapping(&mut self, all_mappings)
            } else if let Some(shj) = plan_any.downcast_ref::<SymmetricHashJoinExec>() {
                let left_size = self.children_nodes[0].plan.schema().fields().len();
                let left_mapping = all_mappings.swap_remove(0);
                let right_mapping = all_mappings.swap_remove(0);
                let new_mapping = left_mapping
                    .iter()
                    .map(|(initial, new)| (initial.clone(), new.clone()))
                    .chain(right_mapping.iter().map(|(initial, new)| {
                        (
                            Column::new(initial.name(), initial.index() + left_size),
                            Column::new(new.name(), new.index() + left_size),
                        )
                    }))
                    .collect::<HashMap<_, _>>();
                self.plan = rewrite_symmetric_hash_join(
                    shj,
                    self.children_nodes[0].plan.clone(),
                    self.children_nodes[1].plan.clone(),
                    &left_mapping,
                    &right_mapping,
                    left_size,
                )?;
                match shj.join_type() {
                    JoinType::Right
                    | JoinType::Full
                    | JoinType::Left
                    | JoinType::Inner => {
                        let (new_left, new_right) =
                            new_mapping.into_iter().partition(|(col_initial, _)| {
                                col_initial.index() < left_size
                            });
                        all_mappings.push(new_left);
                        all_mappings[0].extend(new_right);
                    }
                    JoinType::LeftSemi | JoinType::LeftAnti => {
                        all_mappings.push(left_mapping)
                    }
                    JoinType::RightAnti | JoinType::RightSemi => {
                        all_mappings.push(right_mapping)
                    }
                };
                update_mapping(&mut self, all_mappings)
            } else if let Some(agg) = plan_any.downcast_ref::<AggregateExec>() {
                if agg.aggr_expr().iter().any(|expr| {
                    expr.clone()
                        .with_new_expressions(expr.expressions())
                        .is_none()
                        && !self.children_nodes[0].schema_mapping.is_empty()
                }) {
                    self = preserve_requirements(self)?;
                    return Ok(Transformed::no(self));
                }
                self.plan = if let Some(updated) = rewrite_aggregate(
                    agg,
                    self.children_nodes[0].plan.clone(),
                    &all_mappings[0],
                )? {
                    updated
                } else {
                    return Ok(Transformed::no(self));
                };
            } else if let Some(w_agg) = plan_any.downcast_ref::<WindowAggExec>() {
                self.plan = if let Some(updated) = rewrite_window_aggregate(
                    w_agg,
                    self.children_nodes[0].plan.clone(),
                    &all_mappings[0],
                )? {
                    updated
                } else {
                    return Ok(Transformed::no(self));
                };
                update_mapping(&mut self, all_mappings)
            } else if let Some(bw_agg) = plan_any.downcast_ref::<BoundedWindowAggExec>() {
                self.plan = if let Some(updated) = rewrite_bounded_window_aggregate(
                    bw_agg,
                    self.children_nodes[0].plan.clone(),
                    &all_mappings[0],
                )? {
                    updated
                } else {
                    return Ok(Transformed::no(self));
                };
                update_mapping(&mut self, all_mappings)
            } else if let Some(_file_sink) = plan_any.downcast_ref::<FileSinkExec>() {
                let mapped_exprs =
                    all_mappings.swap_remove(0).into_iter().collect::<Vec<_>>();
                let mut existing_columns =
                    collect_columns_in_plan_schema(&self.children_nodes[0].plan)
                        .into_iter()
                        .collect_vec();
                existing_columns.sort_by_key(|col| col.index());
                let mut exprs = vec![];
                for (idx, _) in existing_columns.iter().enumerate() {
                    if let Some((initial, _final)) = mapped_exprs
                        .iter()
                        .find(|(initial, _final)| initial.index() == idx)
                    {
                        exprs.push((
                            Arc::new(initial.clone()) as Arc<dyn PhysicalExpr>,
                            initial.name().to_string(),
                        ));
                    } else {
                        exprs.push((
                            Arc::new(existing_columns[idx].clone())
                                as Arc<dyn PhysicalExpr>,
                            existing_columns[idx].name().to_string(),
                        ));
                    }
                }
                let projection = Arc::new(ProjectionExec::try_new(
                    exprs,
                    self.children_nodes[0].plan.clone(),
                )?);
                let new_child = ProjectionOptimizer {
                    plan: projection,
                    required_columns: HashSet::new(),
                    schema_mapping: HashMap::new(),
                    children_nodes: vec![self.children_nodes.swap_remove(0)],
                };
                self.plan = self.plan.with_new_children(vec![new_child.plan.clone()])?;
                self.children_nodes = vec![new_child];
            } else {
                return Ok(Transformed::no(self));
            }
        } else {
            let res = self.plan.clone().with_new_children(
                self.children_nodes
                    .iter()
                    .map(|child| child.plan.clone())
                    .collect(),
            );

            self.plan = res?;
        }

        Ok(Transformed::yes(self))
    }

    /// After the top-down pass, there may be some unnecessary projections surviving
    /// since they assumes themselves as necessary when they are analyzed, but after
    /// some optimizations below, they may become unnecessary. This function checks
    /// if the projection is still necessary. If it is not so, it is removed, and
    /// a new mapping is set to the new node, which is the child of the projection,
    /// to transfer the changes resulting from the removal of the projection.
    fn try_remove_projection_bottom_up(mut self) -> Result<Self> {
        let plan = self.plan.clone();
        let Some(projection) = plan.as_any().downcast_ref::<ProjectionExec>() else {
            return Ok(self);
        };

        // Is the projection really required? First, we need to
        // have all column expression in the projection for removal.
        let Some(projection_columns) = try_collect_alias_free_columns(projection.expr())
        else {
            return Ok(self);
        };

        // Then, check if all columns in the input schema exist after
        // the projection. If it is so, we can remove the projection
        // since it does not provide any benefit.
        let child_columns = collect_columns_in_plan_schema(projection.input());
        let child_col_names = child_columns
            .iter()
            .map(|col| col.name().to_string())
            .collect::<HashSet<_>>();
        if child_columns
            .iter()
            .all(|child_col| projection_columns.contains(child_col))
            && child_col_names.len() == child_columns.len()
        {
            // We need to store the existing node's mapping.
            let self_mapping = self.schema_mapping;
            // Remove the projection node.
            self = self.children_nodes.swap_remove(0);

            if self_mapping.is_empty() {
                self.schema_mapping = projection_columns
                    .iter()
                    .enumerate()
                    .filter_map(|(idx, col)| {
                        if col.index() != idx {
                            Some((Column::new(col.name(), idx), col.clone()))
                        } else {
                            None
                        }
                    })
                    .collect();
            } else {
                self.schema_mapping = self_mapping
                    .into_iter()
                    .map(|(expected, updated)| {
                        (
                            expected,
                            Column::new(
                                updated.name(),
                                projection_columns[updated.index()].index(),
                            ),
                        )
                    })
                    .collect()
            }
        }

        Ok(self)
    }
}

impl ConcreteTreeNode for ProjectionOptimizer {
    fn children(&self) -> Vec<&Self> {
        self.children_nodes.iter().collect_vec()
    }

    fn take_children(mut self) -> (Self, Vec<Self>) {
        let children = mem::take(&mut self.children_nodes);
        (self, children)
    }

    fn with_new_children(mut self, children: Vec<Self>) -> Result<Self> {
        self.children_nodes = children;

        self = match self.index_updater()? {
            new_node if new_node.transformed => new_node.data,
            same_node => ProjectionOptimizer::new_default(same_node.data.plan),
        };

        // After the top-down pass, there may be some unnecessary projections surviving
        // since they assumes themselves as necessary when they are analyzed, but after
        // some optimizations below, they may become unnecessary. This check is done
        // here, and if the projection is regarded as unnecessary, the removal would
        // set a new the mapping on the new node, which is the child of the projection.

        self = self.try_remove_projection_bottom_up()?;

        Ok(self)
    }
}

#[derive(Default)]
pub struct OptimizeProjections {}

impl OptimizeProjections {
    #[allow(missing_docs)]
    pub fn new() -> Self {
        Self {}
    }
}

impl PhysicalOptimizerRule for OptimizeProjections {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        // Collect initial columns requirements from the plan's schema.
        let initial_requirements = collect_columns_in_plan_schema(&plan);

        let mut optimizer = ProjectionOptimizer::new_default(plan);

        // Insert the initial requirements to the root node, and run the rule.
        optimizer.required_columns = initial_requirements.clone();
        let mut optimized = optimizer.transform_down(&|o| {
            o.adjust_node_with_requirements().map(Transformed::yes)
        })?;

        // When some projections are removed after the rule, we know that all columns of
        // the initial schema still exist, but their order may be changed. Ensure the final
        // optimized plan satisfies the initial schema order.
        optimized = optimized
            .map_data(|node| satisfy_initial_schema(node, initial_requirements))?;

        Ok(optimized.data.plan)
    }

    fn name(&self) -> &str {
        "OptimizeProjections"
    }

    fn schema_check(&self) -> bool {
        true
    }
}

// If an expression is not trivial and it is referred more than 1,
// unification will not be beneficial as going against caching mechanism
// for non-trivial computations. See the discussion:
// https://github.com/apache/arrow-datafusion/issues/8296
fn caching_projections(
    projection: &ProjectionExec,
    child_projection: &ProjectionExec,
) -> Result<bool> {
    let mut column_ref_map: HashMap<Column, usize> = HashMap::new();
    // Collect the column references' usage in the parent projection.
    projection.expr().iter().try_for_each(|(expr, _)| {
        expr.apply(&mut |expr| {
            Ok({
                if let Some(column) = expr.as_any().downcast_ref::<Column>() {
                    *column_ref_map.entry(column.clone()).or_default() += 1;
                }
                TreeNodeRecursion::Continue
            })
        })?;
        Ok(()) as Result<()>
    })?;
    Ok(column_ref_map.iter().any(|(column, count)| {
        *count > 1 && !is_expr_trivial(&child_projection.expr()[column.index()].0)
    }))
}

/// Ensures that the output schema `po` matches the `initial_requirements`.
/// If the `schema_mapping` of `po` indicates that some columns have been re-mapped,
/// a new projection is added to restore the initial column order and indices.
fn satisfy_initial_schema(
    mut po: ProjectionOptimizer,
    initial_requirements: HashSet<Column>,
) -> Result<ProjectionOptimizer> {
    if po.schema_mapping.is_empty() {
        // The initial schema is already satisfied, no further action required.
        Ok(po)
    } else {
        let mut initial_requirements_ordered =
            initial_requirements.into_iter().collect_vec();
        initial_requirements_ordered.sort_by_key(|expr| expr.index());
        let projected_exprs = initial_requirements_ordered
            .into_iter()
            .map(|col| {
                // If there is a change, get the new index.
                let final_column = po.schema_mapping.remove(&col).unwrap_or(col);
                let final_column_name = final_column.name().to_string();
                let new_col = Arc::new(final_column) as Arc<dyn PhysicalExpr>;
                (new_col, final_column_name)
            })
            .collect::<Vec<_>>();

        // Create the final projection to align with the initial schema.
        let final_projection =
            Arc::new(ProjectionExec::try_new(projected_exprs, po.plan.clone())?);

        // Return a new ProjectionOptimizer with the final projection, resetting the schema mapping.
        Ok(ProjectionOptimizer {
            plan: final_projection,
            required_columns: HashSet::new(),
            schema_mapping: HashMap::new(), // Reset schema mapping as we've now satisfied the initial schema
            children_nodes: vec![po],       // Keep the original node as the child
        })
    }
}

/// Compares the required and existing columns in the node, and maps them accordingly. Caller side must
/// ensure that the node extends its own requirements if the node's plan can introduce new requirements.
fn analyze_requirements(node: &ProjectionOptimizer) -> ColumnRequirements {
    let mut requirement_map = HashMap::new();
    let columns_in_schema = collect_columns_in_plan_schema(&node.plan);
    columns_in_schema.into_iter().for_each(|col| {
        let contains = node.required_columns.contains(&col);
        requirement_map.insert(col, contains);
    });
    requirement_map
}

/// Analyzes the column requirements for join operations between left and right children plans.
///
/// This function compares the required columns from the left and right children with the existing columns in their
/// respective schemas. It determines if there are any redundant fields and creates a mapping to indicate whether
/// each column is required. The function returns a pair of `ColumnRequirements`, one for each child.
///
/// The caller must ensure that the join node extends its requirements if the node's plan can introduce new columns.
/// Each column in the requirement maps corresponds to its own table schema index, not to the join output schema.
///
/// # Arguments
/// * `left_child`: Reference to the execution plan of the left child.
/// * `right_child`: Reference to the execution plan of the right child.
/// * `required_columns`: Set of columns that are required by the parent plan.
/// * `left_size`: Size of the left child's schema, used to adjust the index of right child's columns.
///
/// # Returns
/// A tuple containing two `ColumnRequirements`:
/// - The first element represents the column requirements for the left child.
/// - The second element represents the column requirements for the right child.
///
fn analyze_requirements_of_joins(
    left_child: &Arc<dyn ExecutionPlan>,
    right_child: &Arc<dyn ExecutionPlan>,
    required_columns: &HashSet<Column>,
    left_size: usize,
) -> (ColumnRequirements, ColumnRequirements) {
    let columns_in_schema = collect_columns_in_plan_schema(left_child)
        .into_iter()
        .chain(
            collect_columns_in_plan_schema(right_child)
                .into_iter()
                .map(|col| Column::new(col.name(), col.index() + left_size)),
        );
    let requirement_map = columns_in_schema
        .into_iter()
        .map(|col| {
            if required_columns.contains(&col) {
                (col, true)
            } else {
                (col, false)
            }
        })
        .collect::<HashMap<_, _>>();
    let (requirement_map_left, mut requirement_map_right) =
        requirement_map
            .into_iter()
            .partition::<HashMap<_, _>, _>(|(col, _)| col.index() < left_size);
    requirement_map_right = requirement_map_right
        .into_iter()
        .map(|(col, used)| (Column::new(col.name(), col.index() - left_size), used))
        .collect::<HashMap<_, _>>();
    (requirement_map_left, requirement_map_right)
}

fn preserve_requirements(po: ProjectionOptimizer) -> Result<ProjectionOptimizer> {
    if po.schema_mapping.is_empty() {
        // The initial schema is already satisfied, no further action required.
        Ok(po)
    } else {
        // Collect expressions for the final projection to match the initial requirements.
        let current_fields = collect_columns_in_plan_schema(&po.children_nodes[0].plan);
        let sorted_current_fields = current_fields
            .into_iter()
            .sorted_by_key(|f| f.index())
            .collect::<Vec<_>>();
        let mut projected_exprs = vec![];
        for (idx, field) in po.children_nodes[0]
            .plan
            .schema()
            .fields()
            .iter()
            .enumerate()
        {
            let column = Column::new(field.name(), idx);
            let target = sorted_current_fields[po
                .schema_mapping
                .get(&column)
                .map(|col| col.index())
                .unwrap_or(idx)]
            .clone();
            projected_exprs.push(target);
        }
        let projected_exprs = projected_exprs
            .into_iter()
            .map(|expr| (Arc::new(expr.clone()) as _, expr.name().to_string()))
            .collect::<Vec<_>>();

        // Create the final projection to align with the initial schema.
        let final_projection =
            Arc::new(ProjectionExec::try_new(projected_exprs, po.plan.clone())?);

        // Return a new ProjectionOptimizer with the final projection, resetting the schema mapping.
        Ok(ProjectionOptimizer {
            plan: final_projection,
            required_columns: po.required_columns.clone(),
            schema_mapping: HashMap::new(), // Reset schema mapping as we've now satisfied the initial schema
            children_nodes: vec![po],       // Keep the original node as the child
        })
    }
}

/// Iterates over all columns and returns true if all columns are required.
fn all_columns_required(requirement_map: &ColumnRequirements) -> bool {
    requirement_map.iter().all(|(_k, v)| *v)
}

/// Checks if all columns in the input schema are required by the projection.
///
/// # Arguments
/// * `input_columns`: Reference to a `HashSet<Column>` representing the input columns.
/// * `projection_requires`: Reference to a `HashSet<Column>` representing the projection requirements.
///
/// # Returns
/// `true` if all input columns are required, otherwise `false`.
///
fn all_input_columns_required(
    input_columns: &HashSet<Column>,
    projection_requires: &HashSet<Column>,
) -> bool {
    input_columns
        .iter()
        .all(|input_column| projection_requires.contains(input_column))
}

fn window_agg_required(
    original_schema_len: usize,
    requirements: &ColumnRequirements,
) -> bool {
    requirements
        .iter()
        .filter(|(column, _used)| column.index() >= original_schema_len)
        .any(|(_column, used)| *used)
}

/// Checks if the given expression is trivial.
/// An expression is considered trivial if it is either a `Column` or a `Literal`.
fn is_expr_trivial(expr: &Arc<dyn PhysicalExpr>) -> bool {
    expr.as_any().downcast_ref::<Column>().is_some()
        || expr.as_any().downcast_ref::<Literal>().is_some()
}

/// Compare the inputs and outputs of the projection. All expressions must be
/// columns without alias, and projection does not change the order of fields.
/// For example, if the input schema is `a, b`, `SELECT a, b` is removable,
/// but `SELECT b, a` and `SELECT a+1, b` and `SELECT a AS c, b` are not.
fn is_projection_removable(projection: &ProjectionExec) -> bool {
    let exprs = projection.expr();
    exprs.iter().enumerate().all(|(idx, (expr, alias))| {
        let Some(col) = expr.as_any().downcast_ref::<Column>() else {
            return false;
        };
        col.name() == alias && col.index() == idx
    }) && exprs.len() == projection.input().schema().fields().len()
}

/// Tries to rewrite the [`AggregateExpr`] with the existing expressions to keep on optimization.
fn is_agg_expr_rewritable(aggr_expr: &[Arc<(dyn AggregateExpr)>]) -> bool {
    aggr_expr.iter().all(|expr| expr.expressions().is_empty())
}

/// Tries to rewrite the [`WindowExpr`] with the existing expressions to keep on optimization.
fn is_window_expr_rewritable(window_expr: &[Arc<(dyn WindowExpr)>]) -> bool {
    window_expr.iter().all(|expr| expr.expressions().is_empty())
}

/// Updates a source provider's projected columns according to the given
/// projection operator's expressions. To use this function safely, one must
/// ensure that all expressions are `Column` expressions without aliases.
fn new_projections_for_columns(projection: &[Column], source: &[usize]) -> Vec<usize> {
    projection.iter().map(|col| source[col.index()]).collect()
}

/// When a field in a schema is decided to be redundant and planned to be dropped
/// since it is not required from the plans above, some of the other fields will
/// potentially move to the left side by one. That will change the plans above
/// referring to that field, and they need to update their expressions. This function
/// calculates those index changes and records old and new column expressions in a map.
fn calculate_column_mapping(
    required_columns: &HashSet<Column>,
    unused_columns: &HashSet<Column>,
) -> HashMap<Column, Column> {
    let mut new_mapping = HashMap::new();
    for col in required_columns.iter() {
        let mut skipped_columns = 0;
        for unused_col in unused_columns.iter() {
            if unused_col.index() < col.index() {
                skipped_columns += 1;
            }
        }
        if skipped_columns > 0 {
            new_mapping.insert(
                col.clone(),
                Column::new(col.name(), col.index() - skipped_columns),
            );
        }
    }
    new_mapping
}

/// Given a `ColumnRequirements`, it separates the required and redundant columns.
fn split_column_requirements(
    requirements: &ColumnRequirements,
) -> (HashSet<Column>, HashSet<Column>) {
    let mut required = HashSet::new();
    let mut unused = HashSet::new();
    for (col, is_req) in requirements {
        if *is_req {
            required.insert(col.clone());
        } else {
            unused.insert(col.clone());
        }
    }
    (required, unused)
}

/// Given a set of column expression, constructs a vector having the tuples of `PhysicalExpr`
/// and string alias to be used in creation of `ProjectionExec`. Aliases are the name of columns.
fn convert_projection_exprs(
    columns: HashSet<Column>,
) -> Vec<(Arc<dyn PhysicalExpr>, String)> {
    let mut new_expr = columns.into_iter().collect::<Vec<_>>();
    new_expr.sort_by_key(|column| column.index());
    new_expr
        .into_iter()
        .map(|column| {
            let name = column.name().to_string();
            (Arc::new(column) as Arc<dyn PhysicalExpr>, name)
        })
        .collect()
}

fn extend_left_mapping_with_right(
    mut left_schema_mapping: HashMap<Column, Column>,
    right_child_plan: &Arc<dyn ExecutionPlan>,
    left_size: usize,
    new_left_size: usize,
) -> HashMap<Column, Column> {
    left_schema_mapping.extend(
        right_child_plan
            .schema()
            .fields()
            .iter()
            .enumerate()
            .map(|(idx, field)| {
                (
                    Column::new(field.name(), left_size + idx),
                    Column::new(field.name(), new_left_size + idx),
                )
            })
            .collect::<HashMap<_, _>>(),
    );
    left_schema_mapping
}

/// Calculates the count of removed (unused) columns that precede a given column index.
///
/// # Arguments
/// * `requirement_map`: Reference to a ColumnRequirements map.
/// * `column_index`: The index of the column in question.
///
/// # Returns
/// The number of removed columns before the given column index.
fn removed_column_count(
    requirement_map: &ColumnRequirements,
    column_index: usize,
) -> usize {
    let mut left_skipped_columns = 0;
    for unused_col in
        requirement_map.iter().filter_map(
            |(col, used)| {
                if *used {
                    None
                } else {
                    Some(col)
                }
            },
        )
    {
        if unused_col.index() < column_index {
            left_skipped_columns += 1;
        }
    }
    left_skipped_columns
}

/// Maps the indices of required columns in a parent projection node to the corresponding indices in its child.
///
/// # Example
/// Projection is required to have columns at "@0:a - @1:b - @2:c"
///
/// Projection does "a@2 as a, b@0 as b, c@1 as c"
///
/// Then, projection inserts requirements into its child with these updated indices: "@0:b - @1:c - @2:a"
///
/// # Arguments
/// * `requirements`: Reference to a `HashSet<Column>` representing the parent's column requirements.
/// * `projection_columns`: Slice of `Column` representing the column expressions in the projection.
///
/// # Returns
/// A `HashSet<Column>` with updated column indices reflecting the child's perspective.
///
fn map_parent_reqs_to_input_reqs(
    requirements: &HashSet<Column>,
    projection_columns: &[Column],
) -> HashSet<Column> {
    requirements
        .iter()
        .map(|column| projection_columns[column.index()].clone())
        .collect::<HashSet<_>>()
}

/// Calculates the index changes of columns after the removal of a projection.
///
/// This function iterates through the columns and records the changes in their indices
/// after the removal of a projection. It compares the columns with the columns in the
/// projection and, if a change is observed, maps the old and new index values in a hashmap.
///
/// # Arguments
/// * `columns` - A set of columns before the projection is removed.
/// * `projection_columns` - A slice of columns as they appear in the projection.
///
/// # Returns
/// A `HashMap` where the key is the original column and the value is the column with updated index.
fn index_changes_after_projection_removal(
    columns: HashSet<Column>,
    projection_columns: &[Column],
) -> HashMap<Column, Column> {
    columns
        .into_iter()
        .filter_map(|column| {
            let col_ind = column.index();
            if column != projection_columns[col_ind] {
                Some((column, projection_columns[col_ind].clone()))
            } else {
                None
            }
        })
        .collect()
}

/// Filters the expressions of a [`ProjectionExec`] according to the given used column indices.
fn collect_used_columns(
    projection_exprs: &[(Arc<dyn PhysicalExpr>, String)],
    used_columns: &HashSet<Column>,
) -> Vec<(Arc<dyn PhysicalExpr>, String)> {
    projection_exprs
        .iter()
        .enumerate()
        .filter_map(|(idx, (expr, alias))| {
            if used_columns
                .iter()
                .map(|column| column.index())
                .collect::<Vec<_>>()
                .contains(&idx)
            {
                Some((expr.clone(), alias.clone()))
            } else {
                None
            }
        })
        .collect::<Vec<_>>()
}

fn collect_left_used_columns(
    required_columns: HashSet<Column>,
    left_size: usize,
) -> HashSet<Column> {
    required_columns
        .into_iter()
        .filter(|col| col.index() < left_size)
        .collect()
}

/// Given the expressions of a projection, checks if the projection causes
/// any renaming or constructs a non-`Column` physical expression. If all
/// expressions are `Column`, then they are collected and returned. If not,
/// the function returns `None`.
fn try_collect_alias_free_columns(
    exprs: &[(Arc<dyn PhysicalExpr>, String)],
) -> Option<Vec<Column>> {
    let mut columns = vec![];
    for (expr, alias) in exprs {
        match expr.as_any().downcast_ref::<Column>() {
            Some(column) if column.name() == alias => columns.push(column.clone()),
            _ => return None,
        }
    }
    Some(columns)
}

/// Collects all fields of a schema from a given execution plan and converts them into a [`HashSet`] of [`Column`].
///
/// # Arguments
/// * `plan`: Reference to an Arc of an ExecutionPlan trait object.
///
/// # Returns
/// A `HashSet<Column>` containing all columns from the plan's schema.
fn collect_columns_in_plan_schema(plan: &Arc<dyn ExecutionPlan>) -> HashSet<Column> {
    plan.schema()
        .fields()
        .iter()
        .enumerate()
        .map(|(i, f)| Column::new(f.name(), i))
        .collect()
}

/// Collects all columns involved in the join's equivalence and non-equivalence conditions,
/// adjusting the indices for columns from the right table by adding the size of the left table.
///
/// # Arguments
/// * `on`: Slice of tuples representing the equivalence conditions between columns.
/// * `filter`: Optional reference to a JoinFilter for non-equivalence conditions.
/// * `left_size`: The number of columns in the left table.
/// * `join_left_schema`: Schema reference of the left input to the join.
/// * `join_right_schema`: Schema reference of the right input to the join.
///
/// # Returns
/// A `HashSet<Column>` containing all columns from the join conditions.
fn collect_columns_in_join_conditions(
    on: &[(PhysicalExprRef, PhysicalExprRef)],
    filter: Option<&JoinFilter>,
    left_size: usize,
    join_left_schema: SchemaRef,
    join_right_schema: SchemaRef,
) -> HashSet<Column> {
    let equivalence_columns = on
        .iter()
        .flat_map(|(col_left, col_right)| {
            let left_columns = collect_columns(col_left);
            let right_columns = collect_columns(col_right);
            let right_columns = right_columns
                .into_iter()
                .map(|col| Column::new(col.name(), col.index() + left_size))
                .collect_vec();
            left_columns.into_iter().chain(right_columns).collect_vec()
        })
        .collect::<HashSet<_>>();
    let non_equivalence_columns = filter
        .map(|filter| {
            filter
                .column_indices()
                .iter()
                .map(|col_idx| match col_idx.side {
                    JoinSide::Left => Column::new(
                        join_left_schema.fields()[col_idx.index].name(),
                        col_idx.index,
                    ),
                    JoinSide::Right => Column::new(
                        join_right_schema.fields()[col_idx.index].name(),
                        col_idx.index + left_size,
                    ),
                })
                .collect::<HashSet<_>>()
        })
        .unwrap_or_default();
    equivalence_columns
        .into_iter()
        .chain(non_equivalence_columns)
        .collect::<HashSet<_>>()
}

/// Given expressions of a projection, the function collects all mentioned columns into a vector.
fn collect_column_indices_in_proj_exprs(
    exprs: &[(Arc<dyn PhysicalExpr>, String)],
) -> Vec<usize> {
    exprs
        .iter()
        .flat_map(|(expr, _)| collect_columns(expr))
        .map(|col| col.index())
        .collect::<Vec<_>>()
}

fn split_column_indices(
    file_scan_projection: Vec<usize>,
    required_columns: Vec<usize>,
) -> (Vec<usize>, Vec<usize>) {
    let used_indices = file_scan_projection
        .iter()
        .enumerate()
        .filter_map(|(idx, csv_indx)| {
            if required_columns.contains(&idx) {
                Some(*csv_indx)
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    let unused_indices = file_scan_projection
        .into_iter()
        .enumerate()
        .filter_map(|(idx, csv_idx)| {
            if used_indices.contains(&csv_idx) {
                None
            } else {
                Some(idx)
            }
        })
        .collect::<Vec<_>>();
    (used_indices, unused_indices)
}

fn collect_hj_left_requirements(
    all_requirements: &HashSet<Column>,
    join_projection: &[usize],
    join_left_input_size: usize,
    join_left_schema: SchemaRef,
    on: &[(PhysicalExprRef, PhysicalExprRef)],
    filter: Option<&JoinFilter>,
) -> HashSet<Column> {
    let mut hj_left_requirements = all_requirements
        .iter()
        .filter_map(|req| {
            if join_projection[req.index()] < join_left_input_size {
                Some(Column::new(req.name(), join_projection[req.index()]))
            } else {
                None
            }
        })
        .collect::<HashSet<_>>();
    hj_left_requirements.extend(
        on.iter()
            .flat_map(|(left_on, _)| collect_columns(left_on))
            .collect::<HashSet<_>>(),
    );
    hj_left_requirements.extend(
        filter
            .map(|filter| {
                filter
                    .column_indices()
                    .iter()
                    .filter_map(|col_ind| {
                        if col_ind.side == JoinSide::Left {
                            Some(Column::new(
                                join_left_schema.fields()[col_ind.index].name(),
                                col_ind.index,
                            ))
                        } else {
                            None
                        }
                    })
                    .collect::<HashSet<_>>()
            })
            .unwrap_or_default(),
    );
    hj_left_requirements
}

fn collect_right_hj_left_requirements(
    join_left_schema: SchemaRef,
    on: &[(PhysicalExprRef, PhysicalExprRef)],
    filter: Option<&JoinFilter>,
) -> HashSet<Column> {
    let mut hj_left_requirements = HashSet::new();
    hj_left_requirements.extend(
        on.iter()
            .flat_map(|(left_on, _)| collect_columns(left_on))
            .collect::<HashSet<_>>(),
    );
    hj_left_requirements.extend(
        filter
            .map(|filter| {
                filter
                    .column_indices()
                    .iter()
                    .filter_map(|col_ind| {
                        if col_ind.side == JoinSide::Left {
                            Some(Column::new(
                                join_left_schema.fields()[col_ind.index].name(),
                                col_ind.index,
                            ))
                        } else {
                            None
                        }
                    })
                    .collect::<HashSet<_>>()
            })
            .unwrap_or_default(),
    );
    hj_left_requirements
}

fn collect_hj_right_requirements(
    all_requirements: &HashSet<Column>,
    join_projection: &[usize],
    join_left_input_size: usize,
    join_right_schema: SchemaRef,
    on: &[(PhysicalExprRef, PhysicalExprRef)],
    filter: Option<&JoinFilter>,
) -> HashSet<Column> {
    let mut hj_right_requirements = all_requirements
        .iter()
        .filter_map(|req| {
            if join_projection[req.index()] >= join_left_input_size {
                Some(Column::new(
                    req.name(),
                    join_projection[req.index()] - join_left_input_size,
                ))
            } else {
                None
            }
        })
        .collect::<HashSet<_>>();
    hj_right_requirements.extend(
        on.iter()
            .flat_map(|(_, right_on)| collect_columns(right_on))
            .collect::<HashSet<_>>(),
    );

    hj_right_requirements.extend(
        filter
            .map(|filter| {
                filter
                    .column_indices()
                    .iter()
                    .filter_map(|col_ind| {
                        if col_ind.side == JoinSide::Right {
                            Some(Column::new(
                                join_right_schema.fields()[col_ind.index].name(),
                                col_ind.index,
                            ))
                        } else {
                            None
                        }
                    })
                    .collect::<HashSet<_>>()
            })
            .unwrap_or_default(),
    );
    hj_right_requirements
}

fn collect_right_hj_right_requirements(
    all_requirements: &HashSet<Column>,
    join_projection: &[usize],
    join_right_schema: SchemaRef,
    on: &[(PhysicalExprRef, PhysicalExprRef)],
    filter: Option<&JoinFilter>,
) -> HashSet<Column> {
    let mut hj_right_requirements = all_requirements
        .iter()
        .map(|req| Column::new(req.name(), join_projection[req.index()]))
        .collect::<HashSet<_>>();
    hj_right_requirements.extend(
        on.iter()
            .flat_map(|(_, right_on)| collect_columns(right_on))
            .collect::<HashSet<_>>(),
    );

    hj_right_requirements.extend(
        filter
            .map(|filter| {
                filter
                    .column_indices()
                    .iter()
                    .filter_map(|col_ind| {
                        if col_ind.side == JoinSide::Right {
                            Some(Column::new(
                                join_right_schema.fields()[col_ind.index].name(),
                                col_ind.index,
                            ))
                        } else {
                            None
                        }
                    })
                    .collect::<HashSet<_>>()
            })
            .unwrap_or_default(),
    );
    hj_right_requirements
}

#[derive(Debug, PartialEq)]
enum RewriteState {
    /// The expression is unchanged.
    Unchanged,
    /// Some part of the expression has been rewritten
    RewrittenValid,
    /// Some part of the expression has been rewritten, but some column
    /// references could not be.
    RewrittenInvalid,
}

/// The function operates in two modes:
///
/// 1. When `sync_with_child` is `true`:
///
///    The function updates the indices of `expr` if the expression resides
///    in the projecting input plan. For instance, given the expressions
///    `a@1 + b@2` and `c@0` with the input schema `c@2, a@0, b@1`, the expressions
///    are updated to `a@0 + b@1` and `c@2`.
///
/// 2. When `sync_with_child` is `false`:
///
///    The function determines how the expression would be updated if a projection
///    was placed before the plan associated with the expression. If the expression
///    cannot be rewritten after the projection, it returns `None`. For example,
///    given the expressions `c@0`, `a@1` and `b@2`, and the [`ProjectionExec`] with
///    an output schema of `a, c_new`, then `c@0` becomes `c_new@1`, `a@1` becomes
///    `a@0`, but `b@2` results in `None` since the projection does not include `b`.
fn update_expr_with_projection(
    expr: &Arc<dyn PhysicalExpr>,
    projected_exprs: &[(Arc<dyn PhysicalExpr>, String)],
    sync_with_child: bool,
) -> Result<Option<Arc<dyn PhysicalExpr>>> {
    let mut state = RewriteState::Unchanged;
    let new_expr = expr
        .clone()
        .transform_up_mut(&mut |expr: Arc<dyn PhysicalExpr>| {
            if state == RewriteState::RewrittenInvalid {
                return Ok(Transformed::no(expr));
            }
            let Some(column) = expr.as_any().downcast_ref::<Column>() else {
                return Ok(Transformed::no(expr));
            };
            if sync_with_child {
                state = RewriteState::RewrittenValid;
                // Update the index of `column`:
                Ok(Transformed::yes(projected_exprs[column.index()].0.clone()))
            } else {
                // default to invalid, in case we can't find the relevant column
                state = RewriteState::RewrittenInvalid;
                // Determine how to update `column` to accommodate `projected_exprs`
                projected_exprs
                    .iter()
                    .enumerate()
                    .find_map(|(index, (projected_expr, alias))| {
                        projected_expr.as_any().downcast_ref::<Column>().and_then(
                            |projected_column| {
                                column.eq(projected_column).then(|| {
                                    state = RewriteState::RewrittenValid;
                                    Arc::new(Column::new(alias, index)) as _
                                })
                            },
                        )
                    })
                    .map_or_else(
                        || Ok(Transformed::no(expr)),
                        |c| Ok(Transformed::yes(c)),
                    )
            }
        });
    new_expr.map(|e| (state == RewriteState::RewrittenValid).then_some(e.data))
}

/// Rewrites the expressions with new index values.
fn update_expressions(
    exprs: &[Arc<dyn PhysicalExpr>],
    mapping: &HashMap<Column, Column>,
) -> Vec<Arc<dyn PhysicalExpr>> {
    exprs
        .iter()
        .map(|expr| update_column_index(expr, mapping))
        .collect::<Vec<_>>()
}

/// Rewrites the sort expressions with new index values.
fn update_sort_expressions(
    sort_exprs: &[PhysicalSortExpr],
    mapping: &HashMap<Column, Column>,
) -> LexOrdering {
    sort_exprs
        .iter()
        .map(|sort_expr| PhysicalSortExpr {
            expr: update_column_index(&sort_expr.expr, mapping),
            options: sort_expr.options,
        })
        .collect::<Vec<_>>()
}

/// Updates the expressions subject to hashing of the `Partitioning` according to
/// the mapping. If it is not a hash partitioning, they remains as they are.
fn update_partitioning_expressions(
    partitioning: &Partitioning,
    mapping: &HashMap<Column, Column>,
) -> Partitioning {
    if let Partitioning::Hash(exprs, size) = partitioning {
        let updated_exprs = update_expressions(exprs, mapping);
        Partitioning::Hash(updated_exprs, *size)
    } else {
        partitioning.clone()
    }
}

/// Rewrites the window expressions with new index values.
fn update_window_exprs(
    window_exprs: &[Arc<dyn WindowExpr>],
    mapping: &HashMap<Column, Column>,
) -> Option<Vec<Arc<dyn WindowExpr>>> {
    window_exprs
        .iter()
        .map(|window_expr| {
            let new_exprs = update_expressions(&window_expr.expressions(), mapping);
            window_expr.clone().with_new_expressions(new_exprs)
        })
        .collect::<Option<Vec<_>>>()
}

/// Rewrites the aggregate expressions with new index values.
fn update_aggregate_exprs(
    aggregate_exprs: &[Arc<dyn AggregateExpr>],
    mapping: &HashMap<Column, Column>,
) -> Option<Vec<Arc<dyn AggregateExpr>>> {
    aggregate_exprs
        .iter()
        .map(|aggr_expr| {
            aggr_expr.clone().with_new_expressions(
                aggr_expr
                    .expressions()
                    .iter()
                    .map(|expr| update_column_index(expr, mapping))
                    .collect(),
            )
        })
        .collect::<Option<Vec<_>>>()
}

/// Rewrites the expressions in equivalence condition of a join with new index values.
fn update_join_on(
    join_on: JoinOnRef,
    left_mapping: &HashMap<Column, Column>,
    right_mapping: &HashMap<Column, Column>,
) -> JoinOn {
    join_on
        .iter()
        .map(|(left, right)| {
            (
                update_column_index(left, left_mapping),
                update_column_index(right, right_mapping),
            )
        })
        .collect()
}

/// Given mapping representing the initial and new index values,
/// it updates the indices of columns in the [`PhysicalExpr`].
fn update_column_index(
    expr: &Arc<dyn PhysicalExpr>,
    mapping: &HashMap<Column, Column>,
) -> Arc<dyn PhysicalExpr> {
    let mut state = RewriteState::Unchanged;
    let new_expr = expr
        .clone()
        .transform_up_mut(&mut |expr: Arc<dyn PhysicalExpr>| {
            if state == RewriteState::RewrittenInvalid {
                return Ok(Transformed::no(expr));
            }
            let Some(column) = expr.as_any().downcast_ref::<Column>() else {
                return Ok(Transformed::no(expr));
            };
            state = RewriteState::RewrittenValid;
            // Update the index of `column`:
            if let Some(updated) = mapping.get(column) {
                Ok(Transformed::yes(Arc::new(updated.clone()) as _))
            } else {
                Ok(Transformed::no(expr.clone()))
            }
        })
        .unwrap()
        .data;
    new_expr
}

/// Updates the equivalence conditions of the joins according to the new indices of columns.
fn update_equivalence_conditions(
    on: &[(PhysicalExprRef, PhysicalExprRef)],
    requirement_map_left: &ColumnRequirements,
    requirement_map_right: &ColumnRequirements,
) -> JoinOn {
    on.iter()
        .map(|(left_col, right_col)| {
            let mut left_state = RewriteState::Unchanged;
            let mut right_state = RewriteState::Unchanged;
            (
                left_col
                    .clone()
                    .transform_up_mut(&mut |expr: Arc<dyn PhysicalExpr>| {
                        if left_state == RewriteState::RewrittenInvalid {
                            return Ok(Transformed::no(expr));
                        }
                        let Some(column) = expr.as_any().downcast_ref::<Column>() else {
                            return Ok(Transformed::no(expr));
                        };
                        left_state = RewriteState::RewrittenValid;
                        Ok(Transformed::yes(Arc::new(Column::new(
                            column.name(),
                            column.index()
                                - removed_column_count(
                                    requirement_map_left,
                                    column.index(),
                                ),
                        ))))
                    })
                    .unwrap()
                    .data,
                right_col
                    .clone()
                    .transform_up_mut(&mut |expr: Arc<dyn PhysicalExpr>| {
                        if right_state == RewriteState::RewrittenInvalid {
                            return Ok(Transformed::no(expr));
                        }
                        let Some(column) = expr.as_any().downcast_ref::<Column>() else {
                            return Ok(Transformed::no(expr));
                        };
                        right_state = RewriteState::RewrittenValid;
                        Ok(Transformed::yes(Arc::new(Column::new(
                            column.name(),
                            column.index()
                                - removed_column_count(
                                    requirement_map_right,
                                    column.index(),
                                ),
                        ))))
                    })
                    .unwrap()
                    .data,
            )
        })
        .collect()
}

/// Updates the [`JoinFilter`] according to the new indices of columns.
fn update_non_equivalence_conditions(
    filter: Option<&JoinFilter>,
    requirement_map_left: &ColumnRequirements,
    requirement_map_right: &ColumnRequirements,
) -> Option<JoinFilter> {
    filter.map(|filter| {
        JoinFilter::new(
            filter.expression().clone(),
            filter
                .column_indices()
                .iter()
                .map(|col_idx| match col_idx.side {
                    JoinSide::Left => ColumnIndex {
                        index: col_idx.index
                            - removed_column_count(requirement_map_left, col_idx.index),
                        side: JoinSide::Left,
                    },
                    JoinSide::Right => ColumnIndex {
                        index: col_idx.index
                            - removed_column_count(requirement_map_right, col_idx.index),
                        side: JoinSide::Right,
                    },
                })
                .collect(),
            filter.schema().clone(),
        )
    })
}

fn update_right_child_requirements(
    required_columns: &HashSet<Column>,
    left_size: usize,
) -> HashSet<Column> {
    required_columns
        .iter()
        .filter(|col| col.index() >= left_size)
        .map(|col| Column::new(col.name(), col.index() - left_size))
        .collect()
}

fn update_mapping(
    node: &mut ProjectionOptimizer,
    mut child_mappings: Vec<HashMap<Column, Column>>,
) {
    if node.schema_mapping.is_empty() {
        node.schema_mapping = child_mappings.swap_remove(0);
    } else {
        let child_map = child_mappings.swap_remove(0);
        node.schema_mapping = node
            .schema_mapping
            .iter()
            .map(|(initial, new)| {
                (
                    initial.clone(),
                    child_map.get(new).cloned().unwrap_or(new.clone()),
                )
            })
            .collect()
    }
}

fn update_right_mapping(
    right_schema_mapping: HashMap<Column, Column>,
    left_size: usize,
) -> HashMap<Column, Column> {
    right_schema_mapping
        .into_iter()
        .map(|(old, new)| {
            (
                Column::new(old.name(), old.index() + left_size),
                Column::new(new.name(), new.index() + left_size),
            )
        })
        .collect()
}

fn update_right_requirements(
    required_columns: HashSet<Column>,
    left_size: usize,
) -> HashSet<Column> {
    required_columns
        .into_iter()
        .map(|col| Column::new(col.name(), col.index() + left_size))
        .collect()
}

/// After removal of unused columns in the input table, the columns in projection expression
/// need an update. This function calculates the new positions and updates the indices of columns.
fn update_proj_exprs(
    exprs: &[(Arc<dyn PhysicalExpr>, String)],
    unused_indices: Vec<usize>,
) -> Result<Vec<(Arc<dyn PhysicalExpr>, String)>> {
    exprs
        .iter()
        .map(|(expr, alias)| {
            let new_expr = expr
                .clone()
                .transform_up_mut(&mut |expr: Arc<dyn PhysicalExpr>| {
                    let Some(column) = expr.as_any().downcast_ref::<Column>() else {
                        return Ok(Transformed::no(expr));
                    };
                    let diff = unused_indices
                        .iter()
                        .filter(|idx| **idx < column.index())
                        .count();
                    Ok(Transformed::yes(Arc::new(Column::new(
                        column.name(),
                        column.index() - diff,
                    ))))
                })?
                .data;
            Ok((new_expr, alias.to_owned()))
        })
        .collect::<Result<Vec<_>>>()
}

fn update_hj_children(
    hj_left_requirements: &HashSet<Column>,
    hj_right_requirements: &HashSet<Column>,
    mut children: Vec<ProjectionOptimizer>,
    hj: &HashJoinExec,
) -> Result<(ProjectionOptimizer, ProjectionOptimizer)> {
    let mut ordered_hj_left_requirements =
        hj_left_requirements.iter().cloned().collect_vec();
    ordered_hj_left_requirements.sort_by_key(|col| col.index());
    let left_projection_exprs = ordered_hj_left_requirements
        .into_iter()
        .map(|req| {
            let name = req.name().to_owned();
            (Arc::new(req) as Arc<dyn PhysicalExpr>, name)
        })
        .collect::<Vec<_>>();
    let new_left_projection =
        ProjectionExec::try_new(left_projection_exprs, hj.left().clone())?;
    let new_left_projection_arc = Arc::new(new_left_projection) as Arc<dyn ExecutionPlan>;
    let new_left_requirements = collect_columns_in_plan_schema(&new_left_projection_arc);
    let new_left_node = ProjectionOptimizer {
        plan: new_left_projection_arc,
        required_columns: new_left_requirements,
        schema_mapping: HashMap::new(),
        children_nodes: vec![children.swap_remove(0)],
    };

    let mut ordered_hj_right_requirements =
        hj_right_requirements.iter().cloned().collect_vec();
    ordered_hj_right_requirements.sort_by_key(|col| col.index());
    let right_projection_exprs = ordered_hj_right_requirements
        .into_iter()
        .map(|req| {
            let name = req.name().to_owned();
            (Arc::new(req) as Arc<dyn PhysicalExpr>, name)
        })
        .collect::<Vec<_>>();

    let new_right_projection =
        ProjectionExec::try_new(right_projection_exprs, hj.right().clone())?;
    let new_right_projection_arc =
        Arc::new(new_right_projection) as Arc<dyn ExecutionPlan>;
    let new_right_requirements =
        collect_columns_in_plan_schema(&new_right_projection_arc);
    let new_right_node = ProjectionOptimizer {
        plan: new_right_projection_arc,
        required_columns: new_right_requirements,
        schema_mapping: HashMap::new(),
        children_nodes: vec![children.swap_remove(0)],
    };

    Ok((new_left_node, new_right_node))
}

fn update_hj_left_child(
    hj_left_requirements: &HashSet<Column>,
    hj_right_requirements: &HashSet<Column>,
    mut children: Vec<ProjectionOptimizer>,
    hj: &HashJoinExec,
) -> Result<(ProjectionOptimizer, ProjectionOptimizer)> {
    let mut ordered_hj_left_requirements =
        hj_left_requirements.iter().cloned().collect_vec();
    ordered_hj_left_requirements.sort_by_key(|col| col.index());
    let left_projection_exprs = ordered_hj_left_requirements
        .into_iter()
        .map(|req| {
            let name = req.name().to_owned();
            (Arc::new(req) as Arc<dyn PhysicalExpr>, name)
        })
        .collect::<Vec<_>>();
    let new_left_projection =
        ProjectionExec::try_new(left_projection_exprs, hj.left().clone())?;
    let new_left_projection_arc = Arc::new(new_left_projection) as Arc<dyn ExecutionPlan>;
    let new_left_requirements = collect_columns_in_plan_schema(&new_left_projection_arc);
    let new_left_node = ProjectionOptimizer {
        plan: new_left_projection_arc,
        required_columns: new_left_requirements,
        schema_mapping: HashMap::new(),
        children_nodes: vec![children.swap_remove(0)],
    };

    let mut right_node = children.swap_remove(0);
    right_node.required_columns = hj_right_requirements.clone();

    Ok((new_left_node, right_node))
}

fn update_hj_right_child(
    hj_left_requirements: &HashSet<Column>,
    hj_right_requirements: &HashSet<Column>,
    mut children: Vec<ProjectionOptimizer>,
    hj: &HashJoinExec,
) -> Result<(ProjectionOptimizer, ProjectionOptimizer)> {
    let mut ordered_hj_right_requirements =
        hj_right_requirements.iter().cloned().collect_vec();
    ordered_hj_right_requirements.sort_by_key(|col| col.index());
    let right_projection_exprs = ordered_hj_right_requirements
        .into_iter()
        .map(|req| {
            let name = req.name().to_owned();
            (Arc::new(req) as Arc<dyn PhysicalExpr>, name)
        })
        .collect::<Vec<_>>();
    let new_right_projection =
        ProjectionExec::try_new(right_projection_exprs, hj.right().clone())?;
    let new_right_projection_arc =
        Arc::new(new_right_projection) as Arc<dyn ExecutionPlan>;
    let new_right_requirements =
        collect_columns_in_plan_schema(&new_right_projection_arc);
    let new_right_node = ProjectionOptimizer {
        plan: new_right_projection_arc,
        required_columns: new_right_requirements,
        schema_mapping: HashMap::new(),
        children_nodes: vec![children.swap_remove(1)],
    };

    let mut left_node = children.swap_remove(0);
    left_node.required_columns = hj_left_requirements.clone();

    Ok((left_node, new_right_node))
}

fn update_hj_children_mapping(
    hj_left_requirements: &HashSet<Column>,
    hj_right_requirements: &HashSet<Column>,
) -> (HashMap<Column, Column>, HashMap<Column, Column>) {
    let mut left_mapping = hj_left_requirements.iter().cloned().collect_vec();
    left_mapping.sort_by_key(|col| col.index());
    let left_mapping = left_mapping
        .into_iter()
        .enumerate()
        .map(|(idx, col)| (col.clone(), Column::new(col.name(), idx)))
        .collect::<HashMap<_, _>>();
    let mut right_mapping = hj_right_requirements.iter().collect_vec();
    right_mapping.sort_by_key(|col| col.index());
    let right_mapping = right_mapping
        .into_iter()
        .enumerate()
        .map(|(idx, col)| (col.clone(), Column::new(col.name(), idx)))
        .collect::<HashMap<_, _>>();
    (left_mapping, right_mapping)
}

fn update_hj_projection(
    projection: Option<Vec<usize>>,
    hj_left_schema: SchemaRef,
    hj_left_requirements: HashSet<Column>,
    left_mapping: HashMap<Column, Column>,
    right_mapping: HashMap<Column, Column>,
    join_left_input_size: usize,
) -> Option<Vec<usize>> {
    projection.map(|projection| {
        projection
            .iter()
            .map(|ind| {
                if ind < &hj_left_schema.fields().len() {
                    left_mapping
                        .iter()
                        .find(|(initial, _)| initial.index() == *ind)
                        .map(|(_, target)| target.index())
                        .unwrap_or(*ind)
                } else {
                    right_mapping
                        .iter()
                        .find(|(initial, _)| {
                            initial.index() == *ind - join_left_input_size
                        })
                        .map(|(_, target)| target.index())
                        .unwrap_or(*ind)
                        + hj_left_requirements.len()
                }
            })
            .collect()
    })
}

fn update_hj_projection_right(
    projection: Option<Vec<usize>>,
    right_mapping: HashMap<Column, Column>,
) -> Option<Vec<usize>> {
    projection.map(|projection| {
        projection
            .iter()
            .map(|ind| {
                right_mapping
                    .iter()
                    .find(|(initial, _)| initial.index() == *ind)
                    .map(|(_, target)| target.index())
                    .unwrap_or(*ind)
            })
            .collect()
    })
}

/// Rewrites a filter execution plan with updated column indices.
///
/// This function updates the column indices in a filter's predicate based on a provided mapping.
/// It creates a new `FilterExec` with the updated predicate.
///
/// # Arguments
/// * `predicate` - The predicate expression of the filter.
/// * `input_plan` - The input execution plan on which the filter is applied.
/// * `mapping` - A hashmap with old and new column index mappings.
///
/// # Returns
/// A `Result` containing the new `FilterExec` wrapped in an `Arc`.
fn rewrite_filter(
    predicate: &Arc<dyn PhysicalExpr>,
    input_plan: Arc<dyn ExecutionPlan>,
    mapping: &HashMap<Column, Column>,
) -> Result<Arc<dyn ExecutionPlan>> {
    FilterExec::try_new(update_column_index(predicate, mapping), input_plan)
        .map(|plan| Arc::new(plan) as _)
}

fn rewrite_hj_filter(
    filter: Option<&JoinFilter>,
    left_mapping: &HashMap<Column, Column>,
    right_mapping: &HashMap<Column, Column>,
) -> Option<JoinFilter> {
    filter.map(|filter| {
        JoinFilter::new(
            filter.expression().clone(),
            filter
                .column_indices()
                .iter()
                .map(|col_ind| {
                    if col_ind.side == JoinSide::Left {
                        ColumnIndex {
                            index: left_mapping
                                .iter()
                                .find(|(initial, _)| initial.index() == col_ind.index)
                                .map(|(_, target)| target.index())
                                .unwrap_or(col_ind.index),
                            side: JoinSide::Left,
                        }
                    } else {
                        ColumnIndex {
                            index: right_mapping
                                .iter()
                                .find(|(initial, _)| initial.index() == col_ind.index)
                                .map(|(_, target)| target.index())
                                .unwrap_or(col_ind.index),
                            side: JoinSide::Right,
                        }
                    }
                })
                .collect(),
            filter.schema().clone(),
        )
    })
}

/// Rewrites a projection execution plan with updated column indices.
///
/// This function updates the column indices in a projection based on a provided mapping.
/// It creates a new `ProjectionExec` with the updated expressions.
///
/// # Arguments
/// * `projection` - The original projection execution plan.
/// * `input_plan` - The input execution plan on which the projection is applied.
/// * `mapping` - A hashmap with old and new column index mappings.
///
/// # Returns
/// A `Result` containing the new `ProjectionExec` wrapped in an `Arc`.
///
fn rewrite_projection(
    projection: &ProjectionExec,
    input_plan: Arc<dyn ExecutionPlan>,
    mapping: &HashMap<Column, Column>,
) -> Result<Arc<dyn ExecutionPlan>> {
    ProjectionExec::try_new(
        projection
            .expr()
            .iter()
            .map(|(expr, alias)| (update_column_index(expr, mapping), alias.clone()))
            .collect::<Vec<_>>(),
        input_plan,
    )
    .map(|plan| Arc::new(plan) as _)
}

/// Rewrites a repartition execution plan with updated column indices.
///
/// Updates the partitioning expressions in a repartition plan based on the provided column index mappings.
/// Supports updating the `Partitioning::Hash` variant of partitioning.
///
/// # Arguments
/// * `partitioning` - The original partitioning strategy.
/// * `input_plan` - The input execution plan on which repartitioning is applied.
/// * `mapping` - A hashmap with old and new column index mappings.
///
/// # Returns
/// A `Result` containing the new `RepartitionExec` wrapped in an `Arc`.
fn rewrite_repartition(
    partitioning: &Partitioning,
    input_plan: Arc<dyn ExecutionPlan>,
    mapping: &HashMap<Column, Column>,
) -> Result<Arc<dyn ExecutionPlan>> {
    let new_partitioning = if let Partitioning::Hash(exprs, size) = partitioning {
        let new_exprs = update_expressions(exprs, mapping);
        Partitioning::Hash(new_exprs, *size)
    } else {
        partitioning.clone()
    };
    RepartitionExec::try_new(input_plan, new_partitioning).map(|plan| Arc::new(plan) as _)
}

/// Rewrites a sort execution plan with updated column indices.
///
/// This function updates the column indices in a sort's expressions based on a provided mapping.
/// It creates a new `SortExec` with the updated expressions.
///
/// # Arguments
/// * `sort` - The original sort execution plan.
/// * `input_plan` - The input execution plan on which sorting is applied.
/// * `mapping` - A hashmap with old and new column index mappings.
///
/// # Returns
/// A `Result` containing the new `SortExec` wrapped in an `Arc`.
fn rewrite_sort(
    sort: &SortExec,
    input_plan: Arc<dyn ExecutionPlan>,
    mapping: &HashMap<Column, Column>,
) -> Result<Arc<dyn ExecutionPlan>> {
    let new_sort_exprs = update_sort_expressions(sort.expr(), mapping);
    Ok(Arc::new(
        SortExec::new(new_sort_exprs, input_plan)
            .with_fetch(sort.fetch())
            .with_preserve_partitioning(sort.preserve_partitioning()),
    ) as _)
}

/// Rewrites a sort preserving merge execution plan with updated column indices.
///
/// Updates the sort expressions in a sort preserving merge plan based on the provided column index mappings.
///
/// # Arguments
/// * `sort` - The original `SortPreservingMergeExec` plan.
/// * `input_plan` - The input execution plan to which the sort preserving merge is applied.
/// * `mapping` - A hashmap with old and new column index mappings.
///
/// # Returns
/// A `Result` containing the new `SortPreservingMergeExec` wrapped in an `Arc`.
fn rewrite_sort_preserving_merge(
    sort: &SortPreservingMergeExec,
    input_plan: Arc<dyn ExecutionPlan>,
    mapping: &HashMap<Column, Column>,
) -> Result<Arc<dyn ExecutionPlan>> {
    let new_sort_exprs = update_sort_expressions(sort.expr(), mapping);
    Ok(Arc::new(
        SortPreservingMergeExec::new(new_sort_exprs, input_plan).with_fetch(sort.fetch()),
    ) as _)
}

fn rewrite_nested_loop_join(
    nlj: &NestedLoopJoinExec,
    left_input_plan: Arc<dyn ExecutionPlan>,
    right_input_plan: Arc<dyn ExecutionPlan>,
    left_mapping: &HashMap<Column, Column>,
    right_mapping: &HashMap<Column, Column>,
    left_size: usize,
) -> Result<Arc<dyn ExecutionPlan>> {
    let new_filter = nlj.filter().map(|filter| {
        JoinFilter::new(
            filter.expression().clone(),
            filter
                .column_indices()
                .iter()
                .map(|col_idx| match col_idx.side {
                    JoinSide::Left => ColumnIndex {
                        index: left_mapping
                            .iter()
                            .find(|(old_column, _new_column)| {
                                old_column.index() == col_idx.index
                            })
                            .map(|(_old_column, new_column)| new_column.index())
                            .unwrap_or(col_idx.index),
                        side: JoinSide::Left,
                    },
                    JoinSide::Right => ColumnIndex {
                        index: right_mapping
                            .iter()
                            .find(|(old_column, _new_column)| {
                                old_column.index() == col_idx.index + left_size
                            })
                            .map(|(_old_column, new_column)| new_column.index())
                            .unwrap_or(col_idx.index),
                        side: JoinSide::Left,
                    },
                })
                .collect(),
            filter.schema().clone(),
        )
    });
    NestedLoopJoinExec::try_new(
        left_input_plan,
        right_input_plan,
        new_filter,
        nlj.join_type(),
    )
    .map(|plan| Arc::new(plan) as _)
}

/// Rewrites a sort merge join execution plan.
///
/// This function modifies a SortMergeJoinExec plan with new left and right input plans, updating join conditions and filters according to the provided mappings.
///
/// # Arguments
/// * `smj`: The original SortMergeJoinExec to be rewritten.
/// * `left_input_plan`: The updated execution plan for the left input.
/// * `right_input_plan`: The updated execution plan for the right input.
/// * `left_mapping`: Column mapping for the left input.
/// * `right_mapping`: Column mapping for the right input.
/// * `left_size`: The size of the left input, necessary for index calculations.
///
/// # Returns
/// A Result containing the rewritten execution plan as an Arc, or an error on failure.
fn rewrite_sort_merge_join(
    smj: &SortMergeJoinExec,
    left_input_plan: Arc<dyn ExecutionPlan>,
    right_input_plan: Arc<dyn ExecutionPlan>,
    left_mapping: &HashMap<Column, Column>,
    right_mapping: &HashMap<Column, Column>,
    left_size: usize,
) -> Result<Arc<dyn ExecutionPlan>> {
    let new_on = update_join_on(smj.on(), left_mapping, right_mapping);
    let new_filter = smj.filter.as_ref().map(|filter| {
        JoinFilter::new(
            filter.expression().clone(),
            filter
                .column_indices()
                .iter()
                .map(|col_idx| match col_idx.side {
                    JoinSide::Left => ColumnIndex {
                        index: left_mapping
                            .iter()
                            .find(|(old_column, _new_column)| {
                                old_column.index() == col_idx.index
                            })
                            .map(|(_old_column, new_column)| new_column.index())
                            .unwrap_or(col_idx.index),
                        side: JoinSide::Left,
                    },
                    JoinSide::Right => ColumnIndex {
                        index: right_mapping
                            .iter()
                            .find(|(old_column, _new_column)| {
                                old_column.index() == col_idx.index + left_size
                            })
                            .map(|(_old_column, new_column)| new_column.index())
                            .unwrap_or(col_idx.index),
                        side: JoinSide::Left,
                    },
                })
                .collect(),
            filter.schema().clone(),
        )
    });
    SortMergeJoinExec::try_new(
        left_input_plan,
        right_input_plan,
        new_on,
        new_filter,
        smj.join_type(),
        smj.sort_options.clone(),
        smj.null_equals_null,
    )
    .map(|plan| Arc::new(plan) as _)
}

/// Rewrites a symmetric hash join execution plan.
///
/// Adjusts a SymmetricHashJoinExec plan with new input plans and column mappings, maintaining the original join logic but with updated references.
///
/// # Arguments
/// * `shj`: The SymmetricHashJoinExec to be modified.
/// * `left_input_plan`: New execution plan for the left side.
/// * `right_input_plan`: New execution plan for the right side.
/// * `left_mapping`: Mapping for updating left side columns.
/// * `right_mapping`: Mapping for updating right side columns.
/// * `left_size`: Size of the left input for index adjustments.
///
/// # Returns
/// A Result containing the updated execution plan within an Arc, or an error if the operation fails.
fn rewrite_symmetric_hash_join(
    shj: &SymmetricHashJoinExec,
    left_input_plan: Arc<dyn ExecutionPlan>,
    right_input_plan: Arc<dyn ExecutionPlan>,
    left_mapping: &HashMap<Column, Column>,
    right_mapping: &HashMap<Column, Column>,
    left_size: usize,
) -> Result<Arc<dyn ExecutionPlan>> {
    let new_on = update_join_on(shj.on(), left_mapping, right_mapping);
    let new_filter = shj.filter().map(|filter| {
        JoinFilter::new(
            filter.expression().clone(),
            filter
                .column_indices()
                .iter()
                .map(|col_idx| match col_idx.side {
                    JoinSide::Left => ColumnIndex {
                        index: left_mapping
                            .iter()
                            .find(|(old_column, _new_column)| {
                                old_column.index() == col_idx.index
                            })
                            .map(|(_old_column, new_column)| new_column.index())
                            .unwrap_or(col_idx.index),
                        side: JoinSide::Left,
                    },
                    JoinSide::Right => ColumnIndex {
                        index: right_mapping
                            .iter()
                            .find(|(old_column, _new_column)| {
                                old_column.index() == col_idx.index + left_size
                            })
                            .map(|(_old_column, new_column)| new_column.index())
                            .unwrap_or(col_idx.index),
                        side: JoinSide::Left,
                    },
                })
                .collect(),
            filter.schema().clone(),
        )
    });

    let new_left_sort_exprs = shj.left_sort_exprs().map(|exprs| {
        exprs
            .iter()
            .map(|sort_expr| PhysicalSortExpr {
                expr: update_column_index(&sort_expr.expr, left_mapping),
                options: sort_expr.options,
            })
            .collect()
    });
    let new_right_sort_exprs = shj.left_sort_exprs().map(|exprs| {
        exprs
            .iter()
            .map(|sort_expr| PhysicalSortExpr {
                expr: update_column_index(&sort_expr.expr, right_mapping),
                options: sort_expr.options,
            })
            .collect()
    });
    SymmetricHashJoinExec::try_new(
        left_input_plan,
        right_input_plan,
        new_on,
        new_filter,
        shj.join_type(),
        shj.null_equals_null(),
        new_left_sort_exprs,
        new_right_sort_exprs,
        shj.partition_mode(),
    )
    .map(|plan| Arc::new(plan) as _)
}

/// Rewrites an aggregate execution plan.
///
/// This function updates an AggregateExec plan using a new input plan and column mappings. It adjusts group-by expressions, aggregate expressions, and filters.
///
/// # Arguments
/// * `agg`: The original AggregateExec to be rewritten.
/// * `input_plan`: The new execution plan to use as input.
/// * `mapping`: A mapping from old to new columns.
///
/// # Returns
/// A Result that either contains an Option with the new execution plan wrapped in an Arc, or None if no rewriting is possible, along with an error on failure.
fn rewrite_aggregate(
    agg: &AggregateExec,
    input_plan: Arc<dyn ExecutionPlan>,
    mapping: &HashMap<Column, Column>,
) -> Result<Option<Arc<dyn ExecutionPlan>>> {
    let new_group_by = PhysicalGroupBy::new(
        agg.group_expr()
            .expr()
            .iter()
            .map(|(expr, alias)| (update_column_index(expr, mapping), alias.to_string()))
            .collect(),
        agg.group_expr()
            .null_expr()
            .iter()
            .map(|(expr, alias)| (update_column_index(expr, mapping), alias.to_string()))
            .collect(),
        agg.group_expr().groups().to_vec(),
    );
    let new_agg_expr =
        if let Some(new_agg_expr) = update_aggregate_exprs(agg.aggr_expr(), mapping) {
            new_agg_expr
        } else {
            return Ok(None);
        };
    let new_filter = agg
        .filter_expr()
        .iter()
        .map(|opt_expr| {
            opt_expr
                .clone()
                .map(|expr| update_column_index(&expr, mapping))
        })
        .collect();
    AggregateExec::try_new(
        *agg.mode(),
        new_group_by,
        new_agg_expr,
        new_filter,
        input_plan,
        agg.input_schema(),
    )
    .map(|plan| Some(Arc::new(plan.with_limit(agg.limit())) as _))
}

/// Rewrites a window aggregate execution plan.
///
/// Modifies a WindowAggExec by updating its input plan and expressions based on the provided column mappings, ensuring the window functions are correctly applied to the new plan structure.
///
/// # Arguments
/// * `w_agg`: The WindowAggExec to be updated.
/// * `input_plan`: The new execution plan to be used.
/// * `mapping`: Column mapping for updating window expressions.
///
/// # Returns
/// A Result containing either an Option with the updated execution plan in an Arc, or None if rewriting isn't feasible, and an error on failure.
fn rewrite_window_aggregate(
    w_agg: &WindowAggExec,
    input_plan: Arc<dyn ExecutionPlan>,
    mapping: &HashMap<Column, Column>,
) -> Result<Option<Arc<dyn ExecutionPlan>>> {
    let new_window =
        if let Some(new_window) = update_window_exprs(w_agg.window_expr(), mapping) {
            new_window
        } else {
            return Ok(None);
        };
    let new_partition_keys = update_expressions(&w_agg.partition_keys, mapping);
    WindowAggExec::try_new(new_window, input_plan, new_partition_keys)
        .map(|plan| Some(Arc::new(plan) as _))
}

/// Rewrites a bounded window aggregate execution plan.
///
/// Updates a BoundedWindowAggExec plan with a new input plan and modified window expressions according to provided column mappings, maintaining the logic of bounded window functions.
///
/// # Arguments
/// * `bw_agg`: The original BoundedWindowAggExec to be rewritten.
/// * `input_plan`: The new execution plan to use.
/// * `mapping`: Mapping for updating window expressions.
///
/// # Returns
/// A Result containing an Option with the new execution plan wrapped in an Arc, or None if the rewrite is not possible, and an error on failure.
fn rewrite_bounded_window_aggregate(
    bw_agg: &BoundedWindowAggExec,
    input_plan: Arc<dyn ExecutionPlan>,
    mapping: &HashMap<Column, Column>,
) -> Result<Option<Arc<dyn ExecutionPlan>>> {
    let new_window =
        if let Some(new_window) = update_window_exprs(bw_agg.window_expr(), mapping) {
            new_window
        } else {
            return Ok(None);
        };
    let new_partition_keys = update_expressions(&bw_agg.partition_keys, mapping);
    BoundedWindowAggExec::try_new(
        new_window,
        input_plan,
        new_partition_keys,
        bw_agg.input_order_mode.clone(),
    )
    .map(|plan| Some(Arc::new(plan) as _))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use crate::datasource::file_format::file_compression_type::FileCompressionType;
    use crate::datasource::listing::PartitionedFile;
    use crate::datasource::physical_plan::{CsvExec, FileScanConfig};
    use crate::physical_optimizer::PhysicalOptimizerRule;
    use crate::physical_plan::coalesce_partitions::CoalescePartitionsExec;
    use crate::physical_plan::filter::FilterExec;
    use crate::physical_plan::joins::utils::{ColumnIndex, JoinFilter};
    use crate::physical_plan::joins::StreamJoinPartitionMode;
    use crate::physical_plan::projection::ProjectionExec;
    use crate::physical_plan::repartition::RepartitionExec;
    use crate::physical_plan::sorts::sort::SortExec;
    use crate::physical_plan::sorts::sort_preserving_merge::SortPreservingMergeExec;
    use crate::physical_plan::{get_plan_string, ExecutionPlan};

    use arrow_schema::{DataType, Field, Schema, SortOptions};
    use datafusion_common::config::ConfigOptions;
    use datafusion_common::{JoinSide, JoinType, Result, ScalarValue, Statistics};
    use datafusion_execution::object_store::ObjectStoreUrl;
    use datafusion_expr::{Operator, ScalarFunctionDefinition, WindowFrame};
    use datafusion_physical_expr::expressions::{
        rank, BinaryExpr, CaseExpr, CastExpr, Column, Literal, NegativeExpr, RowNumber,
        Sum,
    };
    use datafusion_physical_expr::window::BuiltInWindowExpr;
    use datafusion_physical_expr::{
        Partitioning, PhysicalExpr, PhysicalSortExpr, ScalarFunctionExpr,
    };
    use datafusion_physical_plan::joins::{
        HashJoinExec, PartitionMode, SymmetricHashJoinExec,
    };
    use datafusion_physical_plan::union::UnionExec;
    use datafusion_physical_plan::InputOrderMode;

    fn create_simple_csv_exec() -> Arc<dyn ExecutionPlan> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, true),
            Field::new("b", DataType::Int32, true),
            Field::new("c", DataType::Int32, true),
            Field::new("d", DataType::Int32, true),
            Field::new("e", DataType::Int32, true),
        ]));

        Arc::new(CsvExec::new(
            FileScanConfig {
                object_store_url: ObjectStoreUrl::parse("test:///").unwrap(),
                file_schema: schema.clone(),
                file_groups: vec![vec![PartitionedFile::new("x".to_string(), 100)]],
                statistics: Statistics::new_unknown(&schema),
                projection: Some(vec![0, 1, 2, 3, 4]),
                limit: None,
                table_partition_cols: vec![],
                output_ordering: vec![vec![]],
            },
            false,
            0,
            0,
            None,
            FileCompressionType::UNCOMPRESSED,
        ))
    }

    fn create_projecting_csv_exec() -> Arc<dyn ExecutionPlan> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, true),
            Field::new("b", DataType::Int32, true),
            Field::new("c", DataType::Int32, true),
            Field::new("d", DataType::Int32, true),
            Field::new("e", DataType::Int32, true),
        ]));

        Arc::new(CsvExec::new(
            FileScanConfig {
                object_store_url: ObjectStoreUrl::parse("test:///").unwrap(),
                file_schema: schema.clone(),
                file_groups: vec![vec![PartitionedFile::new("x".to_string(), 100)]],
                statistics: Statistics::new_unknown(&schema),
                projection: Some(vec![3, 0, 1]),
                limit: None,
                table_partition_cols: vec![],
                output_ordering: vec![vec![]],
            },
            false,
            0,
            0,
            None,
            FileCompressionType::UNCOMPRESSED,
        ))
    }

    #[test]
    fn test_update_matching_exprs() -> Result<()> {
        let exprs: Vec<Arc<dyn PhysicalExpr>> = vec![
            Arc::new(BinaryExpr::new(
                Arc::new(Column::new("a", 3)),
                Operator::Divide,
                Arc::new(Column::new("e", 5)),
            )),
            Arc::new(CastExpr::new(
                Arc::new(Column::new("a", 3)),
                DataType::Float32,
                None,
            )),
            Arc::new(NegativeExpr::new(Arc::new(Column::new("f", 4)))),
            Arc::new(ScalarFunctionExpr::new(
                "scalar_expr",
                ScalarFunctionDefinition::Name(
                    "dummy".to_owned().into_boxed_str().into(),
                ),
                vec![
                    Arc::new(BinaryExpr::new(
                        Arc::new(Column::new("b", 1)),
                        Operator::Divide,
                        Arc::new(Column::new("c", 0)),
                    )),
                    Arc::new(BinaryExpr::new(
                        Arc::new(Column::new("c", 0)),
                        Operator::Divide,
                        Arc::new(Column::new("b", 1)),
                    )),
                ],
                DataType::Int32,
                None,
                false,
            )),
            Arc::new(CaseExpr::try_new(
                Some(Arc::new(Column::new("d", 2))),
                vec![
                    (
                        Arc::new(Column::new("a", 3)) as Arc<dyn PhysicalExpr>,
                        Arc::new(BinaryExpr::new(
                            Arc::new(Column::new("d", 2)),
                            Operator::Plus,
                            Arc::new(Column::new("e", 5)),
                        )) as Arc<dyn PhysicalExpr>,
                    ),
                    (
                        Arc::new(Column::new("a", 3)) as Arc<dyn PhysicalExpr>,
                        Arc::new(BinaryExpr::new(
                            Arc::new(Column::new("e", 5)),
                            Operator::Plus,
                            Arc::new(Column::new("d", 2)),
                        )) as Arc<dyn PhysicalExpr>,
                    ),
                ],
                Some(Arc::new(BinaryExpr::new(
                    Arc::new(Column::new("a", 3)),
                    Operator::Modulo,
                    Arc::new(Column::new("e", 5)),
                ))),
            )?),
        ];
        let child: Vec<(Arc<dyn PhysicalExpr>, String)> = vec![
            (Arc::new(Column::new("c", 2)), "c".to_owned()),
            (Arc::new(Column::new("b", 1)), "b".to_owned()),
            (Arc::new(Column::new("d", 3)), "d".to_owned()),
            (Arc::new(Column::new("a", 0)), "a".to_owned()),
            (Arc::new(Column::new("f", 5)), "f".to_owned()),
            (Arc::new(Column::new("e", 4)), "e".to_owned()),
        ];

        let expected_exprs: Vec<Arc<dyn PhysicalExpr>> = vec![
            Arc::new(BinaryExpr::new(
                Arc::new(Column::new("a", 0)),
                Operator::Divide,
                Arc::new(Column::new("e", 4)),
            )),
            Arc::new(CastExpr::new(
                Arc::new(Column::new("a", 0)),
                DataType::Float32,
                None,
            )),
            Arc::new(NegativeExpr::new(Arc::new(Column::new("f", 5)))),
            Arc::new(ScalarFunctionExpr::new(
                "scalar_expr",
                ScalarFunctionDefinition::Name(
                    "dummy".to_owned().into_boxed_str().into(),
                ),
                vec![
                    Arc::new(BinaryExpr::new(
                        Arc::new(Column::new("b", 1)),
                        Operator::Divide,
                        Arc::new(Column::new("c", 2)),
                    )),
                    Arc::new(BinaryExpr::new(
                        Arc::new(Column::new("c", 2)),
                        Operator::Divide,
                        Arc::new(Column::new("b", 1)),
                    )),
                ],
                DataType::Int32,
                None,
                false,
            )),
            Arc::new(CaseExpr::try_new(
                Some(Arc::new(Column::new("d", 3))),
                vec![
                    (
                        Arc::new(Column::new("a", 0)) as Arc<dyn PhysicalExpr>,
                        Arc::new(BinaryExpr::new(
                            Arc::new(Column::new("d", 3)),
                            Operator::Plus,
                            Arc::new(Column::new("e", 4)),
                        )) as Arc<dyn PhysicalExpr>,
                    ),
                    (
                        Arc::new(Column::new("a", 0)) as Arc<dyn PhysicalExpr>,
                        Arc::new(BinaryExpr::new(
                            Arc::new(Column::new("e", 4)),
                            Operator::Plus,
                            Arc::new(Column::new("d", 3)),
                        )) as Arc<dyn PhysicalExpr>,
                    ),
                ],
                Some(Arc::new(BinaryExpr::new(
                    Arc::new(Column::new("a", 0)),
                    Operator::Modulo,
                    Arc::new(Column::new("e", 4)),
                ))),
            )?),
        ];

        for (expr, expected_expr) in exprs.into_iter().zip(expected_exprs.into_iter()) {
            assert!(update_expr_with_projection(&expr, &child, true)?
                .unwrap()
                .eq(&expected_expr));
        }

        Ok(())
    }

    #[test]
    fn test_update_projected_exprs() -> Result<()> {
        let exprs: Vec<Arc<dyn PhysicalExpr>> = vec![
            Arc::new(BinaryExpr::new(
                Arc::new(Column::new("a", 3)),
                Operator::Divide,
                Arc::new(Column::new("e", 5)),
            )),
            Arc::new(CastExpr::new(
                Arc::new(Column::new("a", 3)),
                DataType::Float32,
                None,
            )),
            Arc::new(NegativeExpr::new(Arc::new(Column::new("f", 4)))),
            Arc::new(ScalarFunctionExpr::new(
                "scalar_expr",
                ScalarFunctionDefinition::Name(
                    "dummy".to_owned().into_boxed_str().into(),
                ),
                vec![
                    Arc::new(BinaryExpr::new(
                        Arc::new(Column::new("b", 1)),
                        Operator::Divide,
                        Arc::new(Column::new("c", 0)),
                    )),
                    Arc::new(BinaryExpr::new(
                        Arc::new(Column::new("c", 0)),
                        Operator::Divide,
                        Arc::new(Column::new("b", 1)),
                    )),
                ],
                DataType::Int32,
                None,
                false,
            )),
            Arc::new(CaseExpr::try_new(
                Some(Arc::new(Column::new("d", 2))),
                vec![
                    (
                        Arc::new(Column::new("a", 3)) as Arc<dyn PhysicalExpr>,
                        Arc::new(BinaryExpr::new(
                            Arc::new(Column::new("d", 2)),
                            Operator::Plus,
                            Arc::new(Column::new("e", 5)),
                        )) as Arc<dyn PhysicalExpr>,
                    ),
                    (
                        Arc::new(Column::new("a", 3)) as Arc<dyn PhysicalExpr>,
                        Arc::new(BinaryExpr::new(
                            Arc::new(Column::new("e", 5)),
                            Operator::Plus,
                            Arc::new(Column::new("d", 2)),
                        )) as Arc<dyn PhysicalExpr>,
                    ),
                ],
                Some(Arc::new(BinaryExpr::new(
                    Arc::new(Column::new("a", 3)),
                    Operator::Modulo,
                    Arc::new(Column::new("e", 5)),
                ))),
            )?),
        ];
        let projected_exprs: Vec<(Arc<dyn PhysicalExpr>, String)> = vec![
            (Arc::new(Column::new("a", 3)), "a".to_owned()),
            (Arc::new(Column::new("b", 1)), "b_new".to_owned()),
            (Arc::new(Column::new("c", 0)), "c".to_owned()),
            (Arc::new(Column::new("d", 2)), "d_new".to_owned()),
            (Arc::new(Column::new("e", 5)), "e".to_owned()),
            (Arc::new(Column::new("f", 4)), "f_new".to_owned()),
        ];

        let expected_exprs: Vec<Arc<dyn PhysicalExpr>> = vec![
            Arc::new(BinaryExpr::new(
                Arc::new(Column::new("a", 0)),
                Operator::Divide,
                Arc::new(Column::new("e", 4)),
            )),
            Arc::new(CastExpr::new(
                Arc::new(Column::new("a", 0)),
                DataType::Float32,
                None,
            )),
            Arc::new(NegativeExpr::new(Arc::new(Column::new("f_new", 5)))),
            Arc::new(ScalarFunctionExpr::new(
                "scalar_expr",
                ScalarFunctionDefinition::Name(
                    "dummy".to_owned().into_boxed_str().into(),
                ),
                vec![
                    Arc::new(BinaryExpr::new(
                        Arc::new(Column::new("b_new", 1)),
                        Operator::Divide,
                        Arc::new(Column::new("c", 2)),
                    )),
                    Arc::new(BinaryExpr::new(
                        Arc::new(Column::new("c", 2)),
                        Operator::Divide,
                        Arc::new(Column::new("b_new", 1)),
                    )),
                ],
                DataType::Int32,
                None,
                false,
            )),
            Arc::new(CaseExpr::try_new(
                Some(Arc::new(Column::new("d_new", 3))),
                vec![
                    (
                        Arc::new(Column::new("a", 0)) as Arc<dyn PhysicalExpr>,
                        Arc::new(BinaryExpr::new(
                            Arc::new(Column::new("d_new", 3)),
                            Operator::Plus,
                            Arc::new(Column::new("e", 4)),
                        )) as Arc<dyn PhysicalExpr>,
                    ),
                    (
                        Arc::new(Column::new("a", 0)) as Arc<dyn PhysicalExpr>,
                        Arc::new(BinaryExpr::new(
                            Arc::new(Column::new("e", 4)),
                            Operator::Plus,
                            Arc::new(Column::new("d_new", 3)),
                        )) as Arc<dyn PhysicalExpr>,
                    ),
                ],
                Some(Arc::new(BinaryExpr::new(
                    Arc::new(Column::new("a", 0)),
                    Operator::Modulo,
                    Arc::new(Column::new("e", 4)),
                ))),
            )?),
        ];

        for (expr, expected_expr) in exprs.into_iter().zip(expected_exprs.into_iter()) {
            assert!(update_expr_with_projection(&expr, &projected_exprs, false)?
                .unwrap()
                .eq(&expected_expr));
        }

        Ok(())
    }

    #[test]
    fn test_csv_after_projection() -> Result<()> {
        let csv = create_projecting_csv_exec();
        let projection: Arc<dyn ExecutionPlan> = Arc::new(ProjectionExec::try_new(
            vec![
                (Arc::new(Column::new("b", 2)), "b".to_string()),
                (Arc::new(Column::new("d", 0)), "d".to_string()),
            ],
            csv.clone(),
        )?);
        let initial = get_plan_string(&projection);
        let expected_initial = [
                "ProjectionExec: expr=[b@2 as b, d@0 as d]",
                "  CsvExec: file_groups={1 group: [[x]]}, projection=[d, a, b], has_header=false",
        ];
        assert_eq!(initial, expected_initial);

        let after_optimize =
            OptimizeProjections::new().optimize(projection, &ConfigOptions::new())?;

        let expected = [
            "CsvExec: file_groups={1 group: [[x]]}, projection=[b, d], has_header=false",
        ];
        assert_eq!(get_plan_string(&after_optimize), expected);

        Ok(())
    }

    #[test]
    fn test_projection_after_projection() -> Result<()> {
        let csv = create_simple_csv_exec();
        let child_projection: Arc<dyn ExecutionPlan> = Arc::new(ProjectionExec::try_new(
            vec![
                (Arc::new(Column::new("c", 2)), "c".to_string()),
                (Arc::new(Column::new("e", 4)), "new_e".to_string()),
                (Arc::new(Column::new("a", 0)), "a".to_string()),
                (Arc::new(Column::new("b", 1)), "new_b".to_string()),
            ],
            csv.clone(),
        )?);
        let top_projection: Arc<dyn ExecutionPlan> = Arc::new(ProjectionExec::try_new(
            vec![
                (Arc::new(Column::new("new_b", 3)), "new_b".to_string()),
                (
                    Arc::new(BinaryExpr::new(
                        Arc::new(Column::new("c", 0)),
                        Operator::Plus,
                        Arc::new(Column::new("new_e", 1)),
                    )),
                    "binary".to_string(),
                ),
                (Arc::new(Column::new("new_b", 3)), "newest_b".to_string()),
            ],
            child_projection.clone(),
        )?);

        let initial = get_plan_string(&top_projection);
        let expected_initial = [
            "ProjectionExec: expr=[new_b@3 as new_b, c@0 + new_e@1 as binary, new_b@3 as newest_b]",
            "  ProjectionExec: expr=[c@2 as c, e@4 as new_e, a@0 as a, b@1 as new_b]",
            "    CsvExec: file_groups={1 group: [[x]]}, projection=[a, b, c, d, e], has_header=false"
            ];
        assert_eq!(initial, expected_initial);

        let after_optimize =
            OptimizeProjections::new().optimize(top_projection, &ConfigOptions::new())?;

        let expected = [
            "ProjectionExec: expr=[b@0 as new_b, c@1 + e@2 as binary, b@0 as newest_b]", 
            "  CsvExec: file_groups={1 group: [[x]]}, projection=[b, c, e], has_header=false"
            ];
        assert_eq!(get_plan_string(&after_optimize), expected);

        Ok(())
    }

    #[test]
    fn test_coalesce_partitions_after_projection() -> Result<()> {
        let csv = create_simple_csv_exec();
        let coalesce_partitions: Arc<dyn ExecutionPlan> =
            Arc::new(CoalescePartitionsExec::new(csv));
        let projection: Arc<dyn ExecutionPlan> = Arc::new(ProjectionExec::try_new(
            vec![
                (Arc::new(Column::new("b", 1)), "b".to_string()),
                (Arc::new(Column::new("a", 0)), "a_new".to_string()),
                (Arc::new(Column::new("d", 3)), "d".to_string()),
            ],
            coalesce_partitions,
        )?);
        let initial = get_plan_string(&projection);
        let expected_initial = [
                "ProjectionExec: expr=[b@1 as b, a@0 as a_new, d@3 as d]",
                "  CoalescePartitionsExec",
                "    CsvExec: file_groups={1 group: [[x]]}, projection=[a, b, c, d, e], has_header=false",
        ];
        assert_eq!(initial, expected_initial);

        let after_optimize =
            OptimizeProjections::new().optimize(projection, &ConfigOptions::new())?;

        let expected = [
                "ProjectionExec: expr=[b@1 as b, a@0 as a_new, d@2 as d]", 
                "  CoalescePartitionsExec", 
                "    CsvExec: file_groups={1 group: [[x]]}, projection=[a, b, d], has_header=false",
        ];
        assert_eq!(get_plan_string(&after_optimize), expected);

        Ok(())
    }

    #[test]
    fn test_filter_after_projection() -> Result<()> {
        let csv = create_simple_csv_exec();
        let predicate = Arc::new(BinaryExpr::new(
            Arc::new(BinaryExpr::new(
                Arc::new(Column::new("b", 1)),
                Operator::Minus,
                Arc::new(Column::new("a", 0)),
            )),
            Operator::Gt,
            Arc::new(BinaryExpr::new(
                Arc::new(Column::new("d", 3)),
                Operator::Minus,
                Arc::new(Column::new("a", 0)),
            )),
        ));
        let filter: Arc<dyn ExecutionPlan> =
            Arc::new(FilterExec::try_new(predicate, csv)?);
        let projection: Arc<dyn ExecutionPlan> = Arc::new(ProjectionExec::try_new(
            vec![
                (Arc::new(Column::new("a", 0)), "a_new".to_string()),
                (Arc::new(Column::new("b", 1)), "b".to_string()),
                (Arc::new(Column::new("d", 3)), "d".to_string()),
            ],
            filter.clone(),
        )?);

        let initial = get_plan_string(&projection);
        let expected_initial = [
                "ProjectionExec: expr=[a@0 as a_new, b@1 as b, d@3 as d]",
                "  FilterExec: b@1 - a@0 > d@3 - a@0",
                "    CsvExec: file_groups={1 group: [[x]]}, projection=[a, b, c, d, e], has_header=false",
        ];
        assert_eq!(initial, expected_initial);

        let after_optimize =
            OptimizeProjections::new().optimize(projection, &ConfigOptions::new())?;

        let expected = [
                "ProjectionExec: expr=[a@0 as a_new, b@1 as b, d@2 as d]", 
                "  FilterExec: b@1 - a@0 > d@2 - a@0", 
                "    CsvExec: file_groups={1 group: [[x]]}, projection=[a, b, d], has_header=false"];

        assert_eq!(get_plan_string(&after_optimize), expected);

        Ok(())
    }

    #[test]
    fn test_join_after_projection() -> Result<()> {
        let left_csv = create_simple_csv_exec();
        let right_csv = create_simple_csv_exec();

        let join: Arc<dyn ExecutionPlan> = Arc::new(SymmetricHashJoinExec::try_new(
            left_csv,
            right_csv,
            vec![(Arc::new(Column::new("b", 1)), Arc::new(Column::new("c", 2)))],
            // b_left-(1+a_right)<=a_right+c_left
            Some(JoinFilter::new(
                Arc::new(BinaryExpr::new(
                    Arc::new(BinaryExpr::new(
                        Arc::new(Column::new("b_left_inter", 0)),
                        Operator::Minus,
                        Arc::new(BinaryExpr::new(
                            Arc::new(Literal::new(ScalarValue::Int32(Some(1)))),
                            Operator::Plus,
                            Arc::new(Column::new("a_right_inter", 1)),
                        )),
                    )),
                    Operator::LtEq,
                    Arc::new(BinaryExpr::new(
                        Arc::new(Column::new("a_right_inter", 1)),
                        Operator::Plus,
                        Arc::new(Column::new("c_left_inter", 2)),
                    )),
                )),
                vec![
                    ColumnIndex {
                        index: 1,
                        side: JoinSide::Left,
                    },
                    ColumnIndex {
                        index: 0,
                        side: JoinSide::Right,
                    },
                    ColumnIndex {
                        index: 2,
                        side: JoinSide::Left,
                    },
                ],
                Schema::new(vec![
                    Field::new("b_left_inter", DataType::Int32, true),
                    Field::new("a_right_inter", DataType::Int32, true),
                    Field::new("c_left_inter", DataType::Int32, true),
                ]),
            )),
            &JoinType::Inner,
            true,
            None,
            None,
            StreamJoinPartitionMode::SinglePartition,
        )?);
        let projection: Arc<dyn ExecutionPlan> = Arc::new(ProjectionExec::try_new(
            vec![
                (Arc::new(Column::new("c", 2)), "c_from_left".to_string()),
                (Arc::new(Column::new("b", 1)), "b_from_left".to_string()),
                (Arc::new(Column::new("a", 0)), "a_from_left".to_string()),
                (Arc::new(Column::new("a", 5)), "a_from_right".to_string()),
                (Arc::new(Column::new("c", 7)), "c_from_right".to_string()),
            ],
            join,
        )?);
        let initial = get_plan_string(&projection);
        let expected_initial = [
            "ProjectionExec: expr=[c@2 as c_from_left, b@1 as b_from_left, a@0 as a_from_left, a@5 as a_from_right, c@7 as c_from_right]", 
            "  SymmetricHashJoinExec: mode=SinglePartition, join_type=Inner, on=[(b@1, c@2)], filter=b_left_inter@0 - 1 + a_right_inter@1 <= a_right_inter@1 + c_left_inter@2", 
            "    CsvExec: file_groups={1 group: [[x]]}, projection=[a, b, c, d, e], has_header=false", 
            "    CsvExec: file_groups={1 group: [[x]]}, projection=[a, b, c, d, e], has_header=false"
            ];
        assert_eq!(initial, expected_initial);

        let after_optimize =
            OptimizeProjections::new().optimize(projection, &ConfigOptions::new())?;

        let expected = [
           "ProjectionExec: expr=[c@2 as c_from_left, b@1 as b_from_left, a@0 as a_from_left, a@3 as a_from_right, c@4 as c_from_right]", 
            "  SymmetricHashJoinExec: mode=SinglePartition, join_type=Inner, on=[(b@1, c@1)], filter=b_left_inter@0 - 1 + a_right_inter@1 <= a_right_inter@1 + c_left_inter@2", 
            "    CsvExec: file_groups={1 group: [[x]]}, projection=[a, b, c], has_header=false", 
            "    CsvExec: file_groups={1 group: [[x]]}, projection=[a, c], has_header=false"
            ];
        assert_eq!(get_plan_string(&after_optimize), expected);

        let expected_filter_col_ind = vec![
            ColumnIndex {
                index: 1,
                side: JoinSide::Left,
            },
            ColumnIndex {
                index: 0,
                side: JoinSide::Right,
            },
            ColumnIndex {
                index: 2,
                side: JoinSide::Left,
            },
        ];

        assert_eq!(
            expected_filter_col_ind,
            after_optimize.children()[0]
                .as_any()
                .downcast_ref::<SymmetricHashJoinExec>()
                .unwrap()
                .filter()
                .unwrap()
                .column_indices()
        );

        Ok(())
    }

    #[test]
    fn test_join_after_required_projection() -> Result<()> {
        let left_csv = create_simple_csv_exec();
        let right_csv = create_simple_csv_exec();

        let join: Arc<dyn ExecutionPlan> = Arc::new(SymmetricHashJoinExec::try_new(
            left_csv,
            right_csv,
            vec![(Arc::new(Column::new("b", 1)), Arc::new(Column::new("c", 2)))],
            // b_left-(1+a_right)<=a_right+c_left
            Some(JoinFilter::new(
                Arc::new(BinaryExpr::new(
                    Arc::new(BinaryExpr::new(
                        Arc::new(Column::new("b_left_inter", 0)),
                        Operator::Minus,
                        Arc::new(BinaryExpr::new(
                            Arc::new(Literal::new(ScalarValue::Int32(Some(1)))),
                            Operator::Plus,
                            Arc::new(Column::new("a_right_inter", 1)),
                        )),
                    )),
                    Operator::LtEq,
                    Arc::new(BinaryExpr::new(
                        Arc::new(Column::new("a_right_inter", 1)),
                        Operator::Plus,
                        Arc::new(Column::new("c_left_inter", 2)),
                    )),
                )),
                vec![
                    ColumnIndex {
                        index: 1,
                        side: JoinSide::Left,
                    },
                    ColumnIndex {
                        index: 0,
                        side: JoinSide::Right,
                    },
                    ColumnIndex {
                        index: 2,
                        side: JoinSide::Left,
                    },
                ],
                Schema::new(vec![
                    Field::new("b_left_inter", DataType::Int32, true),
                    Field::new("a_right_inter", DataType::Int32, true),
                    Field::new("c_left_inter", DataType::Int32, true),
                ]),
            )),
            &JoinType::Inner,
            true,
            None,
            None,
            StreamJoinPartitionMode::SinglePartition,
        )?);
        let projection: Arc<dyn ExecutionPlan> = Arc::new(ProjectionExec::try_new(
            vec![
                (Arc::new(Column::new("a", 5)), "a".to_string()),
                (Arc::new(Column::new("b", 6)), "b".to_string()),
                (Arc::new(Column::new("c", 7)), "c".to_string()),
                (Arc::new(Column::new("d", 8)), "d".to_string()),
                (Arc::new(Column::new("e", 9)), "e".to_string()),
                (Arc::new(Column::new("a", 0)), "a".to_string()),
                (Arc::new(Column::new("b", 1)), "b".to_string()),
                (Arc::new(Column::new("c", 2)), "c".to_string()),
                (Arc::new(Column::new("d", 3)), "d".to_string()),
                (Arc::new(Column::new("e", 4)), "e".to_string()),
            ],
            join,
        )?);
        let initial = get_plan_string(&projection);
        let expected_initial = [
            "ProjectionExec: expr=[a@5 as a, b@6 as b, c@7 as c, d@8 as d, e@9 as e, a@0 as a, b@1 as b, c@2 as c, d@3 as d, e@4 as e]", 
            "  SymmetricHashJoinExec: mode=SinglePartition, join_type=Inner, on=[(b@1, c@2)], filter=b_left_inter@0 - 1 + a_right_inter@1 <= a_right_inter@1 + c_left_inter@2", 
            "    CsvExec: file_groups={1 group: [[x]]}, projection=[a, b, c, d, e], has_header=false", 
            "    CsvExec: file_groups={1 group: [[x]]}, projection=[a, b, c, d, e], has_header=false"
            ];
        assert_eq!(initial, expected_initial);

        let after_optimize =
            OptimizeProjections::new().optimize(projection, &ConfigOptions::new())?;

        let expected = [
            "ProjectionExec: expr=[a@5 as a, b@6 as b, c@7 as c, d@8 as d, e@9 as e, a@0 as a, b@1 as b, c@2 as c, d@3 as d, e@4 as e]", 
            "  SymmetricHashJoinExec: mode=SinglePartition, join_type=Inner, on=[(b@1, c@2)], filter=b_left_inter@0 - 1 + a_right_inter@1 <= a_right_inter@1 + c_left_inter@2", 
            "    CsvExec: file_groups={1 group: [[x]]}, projection=[a, b, c, d, e], has_header=false", 
            "    CsvExec: file_groups={1 group: [[x]]}, projection=[a, b, c, d, e], has_header=false"
            ];
        assert_eq!(get_plan_string(&after_optimize), expected);
        Ok(())
    }

    #[test]
    fn test_hash_join_after_projection() -> Result<()> {
        // sql like
        // SELECT t1.c as c_from_left, t1.b as b_from_left, t1.a as a_from_left, t2.c as c_from_right
        // FROM t1 JOIN t2 ON t1.b = t2.c WHERE t1.b - (1 + t2.a) <= t2.a + t1.c
        let left_csv = create_simple_csv_exec();
        let right_csv = create_simple_csv_exec();

        let join: Arc<dyn ExecutionPlan> = Arc::new(HashJoinExec::try_new(
            left_csv,
            right_csv,
            vec![(Arc::new(Column::new("b", 1)), Arc::new(Column::new("c", 2)))],
            // b_left-(1+a_right)<=a_right+c_left
            Some(JoinFilter::new(
                Arc::new(BinaryExpr::new(
                    Arc::new(BinaryExpr::new(
                        Arc::new(Column::new("b_left_inter", 0)),
                        Operator::Minus,
                        Arc::new(BinaryExpr::new(
                            Arc::new(Literal::new(ScalarValue::Int32(Some(1)))),
                            Operator::Plus,
                            Arc::new(Column::new("a_right_inter", 1)),
                        )),
                    )),
                    Operator::LtEq,
                    Arc::new(BinaryExpr::new(
                        Arc::new(Column::new("a_right_inter", 1)),
                        Operator::Plus,
                        Arc::new(Column::new("c_left_inter", 2)),
                    )),
                )),
                vec![
                    ColumnIndex {
                        index: 1,
                        side: JoinSide::Left,
                    },
                    ColumnIndex {
                        index: 0,
                        side: JoinSide::Right,
                    },
                    ColumnIndex {
                        index: 2,
                        side: JoinSide::Left,
                    },
                ],
                Schema::new(vec![
                    Field::new("b_left_inter", DataType::Int32, true),
                    Field::new("a_right_inter", DataType::Int32, true),
                    Field::new("c_left_inter", DataType::Int32, true),
                ]),
            )),
            &JoinType::Inner,
            None,
            PartitionMode::Auto,
            true,
        )?);
        let projection: Arc<dyn ExecutionPlan> = Arc::new(ProjectionExec::try_new(
            vec![
                (Arc::new(Column::new("c", 2)), "c_from_left".to_string()),
                (Arc::new(Column::new("b", 1)), "b_from_left".to_string()),
                (Arc::new(Column::new("a", 0)), "a_from_left".to_string()),
                (Arc::new(Column::new("c", 7)), "c_from_right".to_string()),
            ],
            join.clone(),
        )?);
        let initial = get_plan_string(&projection);
        let expected_initial = [
			"ProjectionExec: expr=[c@2 as c_from_left, b@1 as b_from_left, a@0 as a_from_left, c@7 as c_from_right]", 
            "  HashJoinExec: mode=Auto, join_type=Inner, on=[(b@1, c@2)], filter=b_left_inter@0 - 1 + a_right_inter@1 <= a_right_inter@1 + c_left_inter@2", 
            "    CsvExec: file_groups={1 group: [[x]]}, projection=[a, b, c, d, e], has_header=false", 
            "    CsvExec: file_groups={1 group: [[x]]}, projection=[a, b, c, d, e], has_header=false"
            ];
        assert_eq!(initial, expected_initial);

        let after_optimize =
            OptimizeProjections::new().optimize(projection, &ConfigOptions::new())?;

        // HashJoinExec only returns result after projection. Because there are some alias columns in the projection, the ProjectionExec is not removed.
        let expected = [
            "ProjectionExec: expr=[c@0 as c_from_left, b@1 as b_from_left, a@2 as a_from_left, c@3 as c_from_right]", 
            "  HashJoinExec: mode=Auto, join_type=Inner, on=[(b@1, c@1)], filter=b_left_inter@0 - 1 + a_right_inter@1 <= a_right_inter@1 + c_left_inter@2, projection=[c@2, b@1, a@0, c@4]", 
            "    CsvExec: file_groups={1 group: [[x]]}, projection=[a, b, c], has_header=false", 
            "    CsvExec: file_groups={1 group: [[x]]}, projection=[a, c], has_header=false"];
        assert_eq!(get_plan_string(&after_optimize), expected);

        let projection: Arc<dyn ExecutionPlan> = Arc::new(ProjectionExec::try_new(
            vec![
                (Arc::new(Column::new("a", 0)), "a".to_string()),
                (Arc::new(Column::new("b", 1)), "b".to_string()),
                (Arc::new(Column::new("c", 2)), "c".to_string()),
                (Arc::new(Column::new("c", 7)), "c".to_string()),
            ],
            join.clone(),
        )?);

        let after_optimize =
            OptimizeProjections::new().optimize(projection, &ConfigOptions::new())?;

        // Comparing to the previous result, this projection don't have alias columns either change the order of output fields. So the ProjectionExec is removed.
        let expected = [
            "HashJoinExec: mode=Auto, join_type=Inner, on=[(b@1, c@1)], filter=b_left_inter@0 - 1 + a_right_inter@1 <= a_right_inter@1 + c_left_inter@2, projection=[a@0, b@1, c@2, c@4]", 
            "  CsvExec: file_groups={1 group: [[x]]}, projection=[a, b, c], has_header=false", 
            "  CsvExec: file_groups={1 group: [[x]]}, projection=[a, c], has_header=false"
            ];
        assert_eq!(get_plan_string(&after_optimize), expected);

        Ok(())
    }

    #[test]
    fn test_repartition_after_projection() -> Result<()> {
        let csv = create_simple_csv_exec();
        let repartition: Arc<dyn ExecutionPlan> = Arc::new(RepartitionExec::try_new(
            csv,
            Partitioning::Hash(
                vec![
                    Arc::new(Column::new("a", 0)),
                    Arc::new(Column::new("b", 1)),
                    Arc::new(Column::new("d", 3)),
                ],
                6,
            ),
        )?);
        let projection: Arc<dyn ExecutionPlan> = Arc::new(ProjectionExec::try_new(
            vec![
                (Arc::new(Column::new("b", 1)), "b_new".to_string()),
                (Arc::new(Column::new("a", 0)), "a".to_string()),
                (Arc::new(Column::new("d", 3)), "d_new".to_string()),
            ],
            repartition,
        )?);
        let initial = get_plan_string(&projection);
        let expected_initial = [
                "ProjectionExec: expr=[b@1 as b_new, a@0 as a, d@3 as d_new]",
                "  RepartitionExec: partitioning=Hash([a@0, b@1, d@3], 6), input_partitions=1",
                "    CsvExec: file_groups={1 group: [[x]]}, projection=[a, b, c, d, e], has_header=false",
        ];
        assert_eq!(initial, expected_initial);

        let after_optimize =
            OptimizeProjections::new().optimize(projection, &ConfigOptions::new())?;
        let expected = [
                "ProjectionExec: expr=[b@1 as b_new, a@0 as a, d@2 as d_new]", 
                "  RepartitionExec: partitioning=Hash([a@0, b@1, d@2], 6), input_partitions=1", 
                "    CsvExec: file_groups={1 group: [[x]]}, projection=[a, b, d], has_header=false"
        ];

        assert_eq!(get_plan_string(&after_optimize), expected);

        Ok(())
    }

    #[test]
    fn test_sort_after_projection() -> Result<()> {
        let csv = create_simple_csv_exec();
        let sort_req: Arc<dyn ExecutionPlan> = Arc::new(SortExec::new(
            vec![
                PhysicalSortExpr {
                    expr: Arc::new(Column::new("b", 1)),
                    options: SortOptions::default(),
                },
                PhysicalSortExpr {
                    expr: Arc::new(BinaryExpr::new(
                        Arc::new(Column::new("c", 2)),
                        Operator::Plus,
                        Arc::new(Column::new("a", 0)),
                    )),
                    options: SortOptions::default(),
                },
            ],
            csv.clone(),
        ));
        let projection: Arc<dyn ExecutionPlan> = Arc::new(ProjectionExec::try_new(
            vec![
                (Arc::new(Column::new("c", 2)), "c".to_string()),
                (Arc::new(Column::new("a", 0)), "new_a".to_string()),
                (Arc::new(Column::new("b", 1)), "b".to_string()),
            ],
            sort_req.clone(),
        )?);

        let initial = get_plan_string(&projection);
        let expected_initial = [
            "ProjectionExec: expr=[c@2 as c, a@0 as new_a, b@1 as b]",
            "  SortExec: expr=[b@1 ASC,c@2 + a@0 ASC]",
            "    CsvExec: file_groups={1 group: [[x]]}, projection=[a, b, c, d, e], has_header=false"
            ];
        assert_eq!(initial, expected_initial);

        let after_optimize =
            OptimizeProjections::new().optimize(projection, &ConfigOptions::new())?;

        let expected = [
            "ProjectionExec: expr=[c@2 as c, a@0 as new_a, b@1 as b]", 
            "  SortExec: expr=[b@1 ASC,c@2 + a@0 ASC]", 
            "    CsvExec: file_groups={1 group: [[x]]}, projection=[a, b, c], has_header=false"
        ];
        assert_eq!(get_plan_string(&after_optimize), expected);

        Ok(())
    }

    #[test]
    fn test_sort_preserving_after_projection() -> Result<()> {
        let csv = create_simple_csv_exec();
        let sort_req: Arc<dyn ExecutionPlan> = Arc::new(SortPreservingMergeExec::new(
            vec![
                PhysicalSortExpr {
                    expr: Arc::new(Column::new("b", 1)),
                    options: SortOptions::default(),
                },
                PhysicalSortExpr {
                    expr: Arc::new(BinaryExpr::new(
                        Arc::new(Column::new("c", 2)),
                        Operator::Plus,
                        Arc::new(Column::new("a", 0)),
                    )),
                    options: SortOptions::default(),
                },
            ],
            csv.clone(),
        ));
        let projection: Arc<dyn ExecutionPlan> = Arc::new(ProjectionExec::try_new(
            vec![
                (Arc::new(Column::new("c", 2)), "c".to_string()),
                (Arc::new(Column::new("a", 0)), "new_a".to_string()),
                (Arc::new(Column::new("b", 1)), "b".to_string()),
            ],
            sort_req.clone(),
        )?);

        let initial = get_plan_string(&projection);
        let expected_initial = [
            "ProjectionExec: expr=[c@2 as c, a@0 as new_a, b@1 as b]",
            "  SortPreservingMergeExec: [b@1 ASC,c@2 + a@0 ASC]",
            "    CsvExec: file_groups={1 group: [[x]]}, projection=[a, b, c, d, e], has_header=false"
            ];
        assert_eq!(initial, expected_initial);

        let after_optimize =
            OptimizeProjections::new().optimize(projection, &ConfigOptions::new())?;

        let expected = [
            "ProjectionExec: expr=[c@2 as c, a@0 as new_a, b@1 as b]", 
            "  SortPreservingMergeExec: [b@1 ASC,c@2 + a@0 ASC]", 
            "    CsvExec: file_groups={1 group: [[x]]}, projection=[a, b, c], has_header=false"
        ];
        assert_eq!(get_plan_string(&after_optimize), expected);

        Ok(())
    }

    #[test]
    fn test_union_after_projection() -> Result<()> {
        let csv = create_simple_csv_exec();
        let union: Arc<dyn ExecutionPlan> =
            Arc::new(UnionExec::new(vec![csv.clone(), csv.clone(), csv]));
        let projection: Arc<dyn ExecutionPlan> = Arc::new(ProjectionExec::try_new(
            vec![
                (Arc::new(Column::new("c", 2)), "c".to_string()),
                (Arc::new(Column::new("a", 0)), "new_a".to_string()),
                (Arc::new(Column::new("b", 1)), "b".to_string()),
            ],
            union.clone(),
        )?);

        let initial = get_plan_string(&projection);
        let expected_initial = [
            "ProjectionExec: expr=[c@2 as c, a@0 as new_a, b@1 as b]", 
            "  UnionExec", 
            "    CsvExec: file_groups={1 group: [[x]]}, projection=[a, b, c, d, e], has_header=false", 
            "    CsvExec: file_groups={1 group: [[x]]}, projection=[a, b, c, d, e], has_header=false", 
            "    CsvExec: file_groups={1 group: [[x]]}, projection=[a, b, c, d, e], has_header=false"
            ];
        assert_eq!(initial, expected_initial);

        let after_optimize =
            OptimizeProjections::new().optimize(projection, &ConfigOptions::new())?;
        let expected = [
            "ProjectionExec: expr=[c@2 as c, a@0 as new_a, b@1 as b]", 
            "  UnionExec", 
            "    CsvExec: file_groups={1 group: [[x]]}, projection=[a, b, c], has_header=false", 
            "    CsvExec: file_groups={1 group: [[x]]}, projection=[a, b, c], has_header=false", 
            "    CsvExec: file_groups={1 group: [[x]]}, projection=[a, b, c], has_header=false"
        ];

        assert_eq!(get_plan_string(&after_optimize), expected);

        Ok(())
    }

    #[test]

    fn test_optimize_projections_filter_sort() -> Result<()> {
        let csv = create_simple_csv_exec();

        let projection1 = Arc::new(ProjectionExec::try_new(
            vec![
                (Arc::new(Column::new("c", 2)), "c".to_string()),
                (Arc::new(Column::new("e", 4)), "x".to_string()),
                (Arc::new(Column::new("a", 0)), "a".to_string()),
            ],
            csv,
        )?);

        let projection2 = Arc::new(ProjectionExec::try_new(
            vec![
                (Arc::new(Column::new("x", 1)), "x".to_string()),
                (Arc::new(Column::new("c", 0)), "c".to_string()),
                (Arc::new(Column::new("a", 2)), "x".to_string()),
            ],
            projection1,
        )?);

        let sort = Arc::new(SortExec::new(
            vec![
                PhysicalSortExpr {
                    expr: Arc::new(Column::new("c", 1)),
                    options: SortOptions::default(),
                },
                PhysicalSortExpr {
                    expr: Arc::new(Column::new("x", 2)),
                    options: SortOptions::default(),
                },
            ],
            projection2,
        ));

        let projection3 = Arc::new(ProjectionExec::try_new(
            vec![
                (Arc::new(Column::new("x", 2)), "x".to_string()),
                (Arc::new(Column::new("x", 0)), "x".to_string()),
                (Arc::new(Column::new("c", 1)), "c".to_string()),
            ],
            sort,
        )?);

        let projection4 = Arc::new(ProjectionExec::try_new(
            vec![(
                Arc::new(BinaryExpr::new(
                    Arc::new(Column::new("c", 2)),
                    Operator::Plus,
                    Arc::new(Column::new("x", 0)),
                )),
                "sum".to_string(),
            )],
            projection3,
        )?);

        let filter = Arc::new(FilterExec::try_new(
            Arc::new(BinaryExpr::new(
                Arc::new(Column::new("sum", 0)),
                Operator::Gt,
                Arc::new(Literal::new(ScalarValue::Int32(Some(0)))),
            )),
            projection4,
        )?) as Arc<dyn ExecutionPlan>;

        let initial = get_plan_string(&filter);

        let expected_initial = [
            "FilterExec: sum@0 > 0", 
            "  ProjectionExec: expr=[c@2 + x@0 as sum]", 
            "    ProjectionExec: expr=[x@2 as x, x@0 as x, c@1 as c]", 
            "      SortExec: expr=[c@1 ASC,x@2 ASC]", 
            "        ProjectionExec: expr=[x@1 as x, c@0 as c, a@2 as x]", 
            "          ProjectionExec: expr=[c@2 as c, e@4 as x, a@0 as a]", 
            "            CsvExec: file_groups={1 group: [[x]]}, projection=[a, b, c, d, e], has_header=false"];

        assert_eq!(initial, expected_initial);

        let after_optimize =
            OptimizeProjections::new().optimize(filter, &ConfigOptions::new())?;

        let expected = [
            "FilterExec: sum@0 > 0", 
            "  ProjectionExec: expr=[c@0 + x@1 as sum]", 
            "    SortExec: expr=[c@0 ASC,x@1 ASC]", 
            "      ProjectionExec: expr=[c@1 as c, a@0 as x]", 
            "        CsvExec: file_groups={1 group: [[x]]}, projection=[a, c], has_header=false"];

        assert_eq!(get_plan_string(&after_optimize), expected);
        Ok(())
    }

    #[test]
    fn test_optimize_projections_left_anti() -> Result<()> {
        let csv_left = create_simple_csv_exec();
        let window = Arc::new(WindowAggExec::try_new(
            vec![Arc::new(BuiltInWindowExpr::new(
                Arc::new(RowNumber::new("ROW_NUMBER()".to_string(), &DataType::Int32)),
                &[],
                &[],
                Arc::new(WindowFrame::new(None)),
            ))],
            csv_left,
            vec![],
        )?);
        let projection = Arc::new(ProjectionExec::try_new(
            vec![
                (
                    Arc::new(Column::new("ROW_NUMBER()", 5)),
                    "ROW_NUMBER()".to_string(),
                ),
                (Arc::new(Column::new("a", 0)), "a".to_string()),
                (Arc::new(Column::new("d", 3)), "d".to_string()),
                (Arc::new(Column::new("d", 3)), "d".to_string()),
                (
                    Arc::new(Column::new("ROW_NUMBER()", 5)),
                    "ROW_NUMBER()".to_string(),
                ),
            ],
            window,
        )?);
        let sort = Arc::new(SortExec::new(
            vec![
                PhysicalSortExpr {
                    expr: Arc::new(Column::new("ROW_NUMBER()", 4)),
                    options: SortOptions::default(),
                },
                PhysicalSortExpr {
                    expr: Arc::new(Column::new("d", 2)),
                    options: SortOptions::default(),
                },
            ],
            projection,
        ));

        let csv_right = create_simple_csv_exec();
        let bounded_window = Arc::new(BoundedWindowAggExec::try_new(
            vec![Arc::new(BuiltInWindowExpr::new(
                Arc::new(rank("RANK()".to_string(), &DataType::Int32)),
                &[],
                &[],
                Arc::new(WindowFrame::new(None)),
            ))],
            csv_right,
            vec![],
            InputOrderMode::Linear,
        )?);

        let join = Arc::new(HashJoinExec::try_new(
            sort,
            bounded_window,
            vec![(Arc::new(Column::new("a", 1)), Arc::new(Column::new("b", 1)))],
            Some(JoinFilter::new(
                Arc::new(BinaryExpr::new(
                    Arc::new(Column::new("ROW_NUMBER()", 0)),
                    Operator::Lt,
                    Arc::new(Column::new("RANK", 1)),
                )),
                vec![
                    ColumnIndex {
                        index: 4,
                        side: JoinSide::Left,
                    },
                    ColumnIndex {
                        index: 5,
                        side: JoinSide::Right,
                    },
                ],
                Schema::new(vec![
                    Field::new("inter_rownumber", DataType::Int32, true),
                    Field::new("inter_avg_b", DataType::Float32, true),
                ]),
            )),
            &JoinType::LeftAnti,
            Some(vec![1, 0, 1]),
            PartitionMode::Auto,
            true,
        )?);
        let aggregate = Arc::new(AggregateExec::try_new(
            AggregateMode::Single,
            PhysicalGroupBy::new(
                vec![(Arc::new(Column::new("a", 2)), "a".to_owned())],
                vec![],
                vec![],
            ),
            vec![Arc::new(Sum::new(
                Arc::new(Column::new("ROW_NUMBER()", 1)),
                "SUM(ROW_NUMBER())",
                DataType::Int64,
            ))],
            vec![None],
            join.clone(),
            join.schema(),
        )?) as Arc<dyn ExecutionPlan>;

        let initial = get_plan_string(&aggregate);

        let expected_initial = [
            "AggregateExec: mode=Single, gby=[a@2 as a], aggr=[SUM(ROW_NUMBER())]", 
            "  HashJoinExec: mode=Auto, join_type=LeftAnti, on=[(a@1, b@1)], filter=ROW_NUMBER()@0 < RANK@1, projection=[a@1, ROW_NUMBER()@0, a@1]", 
            "    SortExec: expr=[ROW_NUMBER()@4 ASC,d@2 ASC]", 
            "      ProjectionExec: expr=[ROW_NUMBER()@5 as ROW_NUMBER(), a@0 as a, d@3 as d, d@3 as d, ROW_NUMBER()@5 as ROW_NUMBER()]", 
            "        WindowAggExec: wdw=[ROW_NUMBER(): Ok(Field { name: \"ROW_NUMBER()\", data_type: Int32, nullable: false, dict_id: 0, dict_is_ordered: false, metadata: {} }), frame: WindowFrame { units: Rows, start_bound: Preceding(UInt64(NULL)), end_bound: Following(UInt64(NULL)), is_causal: false }]", 
            "          CsvExec: file_groups={1 group: [[x]]}, projection=[a, b, c, d, e], has_header=false", 
            "    BoundedWindowAggExec: wdw=[RANK(): Ok(Field { name: \"RANK()\", data_type: Int32, nullable: false, dict_id: 0, dict_is_ordered: false, metadata: {} }), frame: WindowFrame { units: Rows, start_bound: Preceding(UInt64(NULL)), end_bound: Following(UInt64(NULL)), is_causal: false }], mode=[Linear]", 
            "      CsvExec: file_groups={1 group: [[x]]}, projection=[a, b, c, d, e], has_header=false"];

        assert_eq!(initial, expected_initial);

        let after_optimize =
            OptimizeProjections::new().optimize(aggregate, &ConfigOptions::new())?;

        let expected = [
            "AggregateExec: mode=Single, gby=[a@2 as a], aggr=[SUM(ROW_NUMBER())]", 
            "  HashJoinExec: mode=Auto, join_type=LeftAnti, on=[(a@1, b@0)], filter=ROW_NUMBER()@0 < RANK@1, projection=[a@1, ROW_NUMBER()@0, a@1]", 
            "    ProjectionExec: expr=[ROW_NUMBER()@0 as ROW_NUMBER(), a@1 as a, ROW_NUMBER()@3 as ROW_NUMBER()]", 
            "      SortExec: expr=[ROW_NUMBER()@3 ASC,d@2 ASC]", 
            "        ProjectionExec: expr=[ROW_NUMBER()@5 as ROW_NUMBER(), a@0 as a, d@3 as d, ROW_NUMBER()@5 as ROW_NUMBER()]", 
            "          WindowAggExec: wdw=[ROW_NUMBER(): Ok(Field { name: \"ROW_NUMBER()\", data_type: Int32, nullable: false, dict_id: 0, dict_is_ordered: false, metadata: {} }), frame: WindowFrame { units: Rows, start_bound: Preceding(UInt64(NULL)), end_bound: Following(UInt64(NULL)), is_causal: false }]", 
            "            CsvExec: file_groups={1 group: [[x]]}, projection=[a, b, c, d, e], has_header=false", 
            "    ProjectionExec: expr=[b@1 as b, RANK()@5 as RANK()]", 
            "      BoundedWindowAggExec: wdw=[RANK(): Ok(Field { name: \"RANK()\", data_type: Int32, nullable: false, dict_id: 0, dict_is_ordered: false, metadata: {} }), frame: WindowFrame { units: Rows, start_bound: Preceding(UInt64(NULL)), end_bound: Following(UInt64(NULL)), is_causal: false }], mode=[Linear]", 
            "        CsvExec: file_groups={1 group: [[x]]}, projection=[a, b, c, d, e], has_header=false"];

        assert_eq!(get_plan_string(&after_optimize), expected);

        Ok(())
    }

    #[test]
    fn test_optimize_projections_right_semi() -> Result<()> {
        let csv_left = create_simple_csv_exec();
        let window = Arc::new(WindowAggExec::try_new(
            vec![Arc::new(BuiltInWindowExpr::new(
                Arc::new(RowNumber::new("ROW_NUMBER()".to_string(), &DataType::Int32)),
                &[],
                &[],
                Arc::new(WindowFrame::new(None)),
            ))],
            csv_left,
            vec![],
        )?);
        let projection = Arc::new(ProjectionExec::try_new(
            vec![
                (
                    Arc::new(Column::new("ROW_NUMBER()", 5)),
                    "ROW_NUMBER()".to_string(),
                ),
                (Arc::new(Column::new("a", 0)), "a".to_string()),
                (Arc::new(Column::new("d", 3)), "d".to_string()),
                (Arc::new(Column::new("d", 3)), "d".to_string()),
                (
                    Arc::new(Column::new("ROW_NUMBER()", 5)),
                    "ROW_NUMBER()".to_string(),
                ),
            ],
            window,
        )?);
        let sort = Arc::new(SortExec::new(
            vec![
                PhysicalSortExpr {
                    expr: Arc::new(Column::new("ROW_NUMBER()", 4)),
                    options: SortOptions::default(),
                },
                PhysicalSortExpr {
                    expr: Arc::new(Column::new("d", 2)),
                    options: SortOptions::default(),
                },
            ],
            projection,
        ));

        let csv_right = create_simple_csv_exec();
        let bounded_window = Arc::new(BoundedWindowAggExec::try_new(
            vec![Arc::new(BuiltInWindowExpr::new(
                Arc::new(rank("RANK()".to_string(), &DataType::Int32)),
                &[],
                &[],
                Arc::new(WindowFrame::new(None)),
            ))],
            csv_right,
            vec![],
            InputOrderMode::Linear,
        )?);

        let join = Arc::new(HashJoinExec::try_new(
            sort,
            bounded_window,
            vec![(Arc::new(Column::new("a", 1)), Arc::new(Column::new("b", 1)))],
            Some(JoinFilter::new(
                Arc::new(BinaryExpr::new(
                    Arc::new(Column::new("ROW_NUMBER()", 0)),
                    Operator::Lt,
                    Arc::new(Column::new("RANK", 1)),
                )),
                vec![
                    ColumnIndex {
                        index: 4,
                        side: JoinSide::Left,
                    },
                    ColumnIndex {
                        index: 5,
                        side: JoinSide::Right,
                    },
                ],
                Schema::new(vec![
                    Field::new("inter_rownumber", DataType::Int32, true),
                    Field::new("inter_avg_b", DataType::Float32, true),
                ]),
            )),
            &JoinType::RightSemi,
            Some(vec![1, 0, 1]),
            PartitionMode::Auto,
            true,
        )?);
        let aggregate = Arc::new(AggregateExec::try_new(
            AggregateMode::Single,
            PhysicalGroupBy::new(
                vec![(Arc::new(Column::new("b", 2)), "b".to_owned())],
                vec![],
                vec![],
            ),
            vec![Arc::new(Sum::new(
                Arc::new(Column::new("a", 1)),
                "SUM(a)",
                DataType::Int64,
            ))],
            vec![None],
            join.clone(),
            join.schema(),
        )?) as Arc<dyn ExecutionPlan>;

        let initial = get_plan_string(&aggregate);

        let expected_initial = [
            "AggregateExec: mode=Single, gby=[b@2 as b], aggr=[SUM(a)]", 
            "  HashJoinExec: mode=Auto, join_type=RightSemi, on=[(a@1, b@1)], filter=ROW_NUMBER()@0 < RANK@1, projection=[b@1, a@0, b@1]", 
            "    SortExec: expr=[ROW_NUMBER()@4 ASC,d@2 ASC]", 
            "      ProjectionExec: expr=[ROW_NUMBER()@5 as ROW_NUMBER(), a@0 as a, d@3 as d, d@3 as d, ROW_NUMBER()@5 as ROW_NUMBER()]", 
            "        WindowAggExec: wdw=[ROW_NUMBER(): Ok(Field { name: \"ROW_NUMBER()\", data_type: Int32, nullable: false, dict_id: 0, dict_is_ordered: false, metadata: {} }), frame: WindowFrame { units: Rows, start_bound: Preceding(UInt64(NULL)), end_bound: Following(UInt64(NULL)), is_causal: false }]", 
            "          CsvExec: file_groups={1 group: [[x]]}, projection=[a, b, c, d, e], has_header=false", 
            "    BoundedWindowAggExec: wdw=[RANK(): Ok(Field { name: \"RANK()\", data_type: Int32, nullable: false, dict_id: 0, dict_is_ordered: false, metadata: {} }), frame: WindowFrame { units: Rows, start_bound: Preceding(UInt64(NULL)), end_bound: Following(UInt64(NULL)), is_causal: false }], mode=[Linear]", 
            "      CsvExec: file_groups={1 group: [[x]]}, projection=[a, b, c, d, e], has_header=false"];

        assert_eq!(initial, expected_initial);

        let after_optimize =
            OptimizeProjections::new().optimize(aggregate, &ConfigOptions::new())?;

        let expected = [
            "AggregateExec: mode=Single, gby=[b@2 as b], aggr=[SUM(a)]", 
            "  HashJoinExec: mode=Auto, join_type=RightSemi, on=[(a@0, b@1)], filter=ROW_NUMBER()@0 < RANK@1, projection=[b@1, a@0, b@1]", 
            "    ProjectionExec: expr=[a@0 as a, ROW_NUMBER()@2 as ROW_NUMBER()]", 
            "      SortExec: expr=[ROW_NUMBER()@2 ASC,d@1 ASC]", 
            "        ProjectionExec: expr=[a@0 as a, d@3 as d, ROW_NUMBER()@5 as ROW_NUMBER()]", 
            "          WindowAggExec: wdw=[ROW_NUMBER(): Ok(Field { name: \"ROW_NUMBER()\", data_type: Int32, nullable: false, dict_id: 0, dict_is_ordered: false, metadata: {} }), frame: WindowFrame { units: Rows, start_bound: Preceding(UInt64(NULL)), end_bound: Following(UInt64(NULL)), is_causal: false }]", 
            "            CsvExec: file_groups={1 group: [[x]]}, projection=[a, b, c, d, e], has_header=false", 
            "    ProjectionExec: expr=[a@0 as a, b@1 as b, RANK()@5 as RANK()]", 
            "      BoundedWindowAggExec: wdw=[RANK(): Ok(Field { name: \"RANK()\", data_type: Int32, nullable: false, dict_id: 0, dict_is_ordered: false, metadata: {} }), frame: WindowFrame { units: Rows, start_bound: Preceding(UInt64(NULL)), end_bound: Following(UInt64(NULL)), is_causal: false }], mode=[Linear]", 
            "        CsvExec: file_groups={1 group: [[x]]}, projection=[a, b, c, d, e], has_header=false"];

        assert_eq!(get_plan_string(&after_optimize), expected);

        Ok(())
    }
}
