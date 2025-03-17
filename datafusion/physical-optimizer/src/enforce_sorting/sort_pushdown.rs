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

use std::fmt::Debug;
use std::sync::Arc;

use crate::utils::{
    add_sort_above, is_sort, is_sort_preserving_merge, is_union, is_window,
};

use arrow::datatypes::SchemaRef;
use datafusion_common::tree_node::{
    ConcreteTreeNode, Transformed, TreeNode, TreeNodeRecursion,
};
use datafusion_common::{plan_err, HashSet, JoinSide, Result};
use datafusion_expr::JoinType;
use datafusion_physical_expr::expressions::Column;
use datafusion_physical_expr::utils::collect_columns;
use datafusion_physical_expr::PhysicalSortRequirement;
use datafusion_physical_expr_common::sort_expr::{LexOrdering, LexRequirement};
use datafusion_physical_plan::execution_plan::RequiredInputOrdering;
use datafusion_physical_plan::filter::FilterExec;
use datafusion_physical_plan::joins::utils::{
    calculate_join_output_ordering, ColumnIndex,
};
use datafusion_physical_plan::joins::{HashJoinExec, SortMergeJoinExec};
use datafusion_physical_plan::projection::ProjectionExec;
use datafusion_physical_plan::repartition::RepartitionExec;
use datafusion_physical_plan::sorts::sort::SortExec;
use datafusion_physical_plan::tree_node::PlanContext;
use datafusion_physical_plan::windows::BoundedWindowAggExec;
use datafusion_physical_plan::{ExecutionPlan, ExecutionPlanProperties, InputOrderMode};

/// This is a "data class" we use within the [`EnforceSorting`] rule to push
/// down [`SortExec`] in the plan. In some cases, we can reduce the total
/// computational cost by pushing down `SortExec`s through some executors. The
/// object carries the parent required ordering and the (optional) `fetch` value
/// of the parent node as its data.
///
/// [`EnforceSorting`]: crate::enforce_sorting::EnforceSorting
#[derive(Default, Clone, Debug)]
pub struct ParentRequirements {
    /// The required input ordering. If a [`SortExec`] is removed we turn its information as a state
    /// so that we can re-add at the lowest possible level
    ordering_requirement: Option<RequiredInputOrdering>,
    /// Fetch information about the removed [`SortExec`]
    fetch: Option<usize>,
    /// The plans that caused a soft requirement. This information is being used in `to_linear_plans` when necessary
    soft_requirement_plans: Vec<Arc<dyn ExecutionPlan>>,
    /// At some cases, we softly require input ordering so that the operators can work with Sorted input order modes.
    /// If we decide to remove the [`SortExec`]s that satisfy these soft requirements,
    /// we need to know which plans must turn into Linear input order modes. to_linear_plans holds these execution plans' Arcs.
    to_linear_plans: Vec<Arc<dyn ExecutionPlan>>,
    /// State information for RequiredInputOrdering types.
    /// Identifies if children can be turned into soft requirements or not.
    /// If a Fetch information is encountered, keep_hard flag will be true with no specification,
    /// that means all Sorts needs to be preserved
    /// Or if a Hard Sort requirement is above, children Sorts can only be set to Soft if they're not compatible with parent Sort
    keep_hard_requirement: KeepHardRequirement,
    /// TODO doc
    hard_parent_requirement: KeepHardRequirement,
}

#[derive(Default, Clone, Debug)]
struct KeepHardRequirement {
    /// If set, specifies the parent ordering to protect the requirement
    /// If not set acts as a wildcard, protects every requirement
    specification: Option<RequiredInputOrdering>,
    /// Should we keep the ordering requirement as Hard?
    keep_hard: bool,
}

impl KeepHardRequirement {
    /// Check if given requirements satisfies the plan, if so return Hard requirement or else Soft.
    fn check_based_on_specification(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        child_req: RequiredInputOrdering,
    ) -> RequiredInputOrdering {
        if let Some(spec) = self.specification.clone() {
            if plan
                .equivalence_properties()
                .ordering_satisfy_requirement(spec.lex_requirement())
            {
                RequiredInputOrdering::Hard(LexRequirement::new(child_req.to_vec()))
            } else {
                RequiredInputOrdering::Soft(LexRequirement::new(child_req.to_vec()))
            }
        } else {
            RequiredInputOrdering::Hard(LexRequirement::new(child_req.to_vec()))
        }
    }
}

pub type SortPushDown = PlanContext<ParentRequirements>;

/// Assigns the ordering requirement of the root node to its children.
pub fn assign_initial_requirements(sort_push_down: &mut SortPushDown) {
    let reqs = sort_push_down.plan.required_input_ordering();
    for (child, requirement) in sort_push_down.children.iter_mut().zip(reqs) {
        child.data = ParentRequirements {
            ordering_requirement: requirement,
            // If the parent has a fetch value, assign it to the children
            // Or use the fetch value of the child.
            fetch: child.plan.fetch(),
            soft_requirement_plans: vec![],
            to_linear_plans: vec![],
            keep_hard_requirement: KeepHardRequirement::default(),
            hard_parent_requirement: KeepHardRequirement::default(),
        };
    }
}

/// Tries to push down the sort requirements as far as possible, if decides a `SortExec` is unnecessary removes it.
pub fn pushdown_sorts(sort_push_down: SortPushDown) -> Result<SortPushDown> {
    // TODO Transform pushdown'a çevirince bu to_linear_plans stratejisi tutmadı, planın çocukları değişmiş halde geldiği için eşitlik sağlanmıyor
    let mut sort_push_down = pushdown_sorts_helper(sort_push_down)?;
    while sort_push_down.tnr == TreeNodeRecursion::Stop {
        sort_push_down = pushdown_sorts_helper(sort_push_down.data)?;
    }

    // TODO Burayı da transform pushdown'a uygun hale getirelim.
    let (new_node, children) = sort_push_down.data.take_children();
    let new_children = children
        .into_iter()
        .map(pushdown_sorts)
        .collect::<Result<Vec<SortPushDown>>>()?;

    let mut linear_plans = vec![];
    for new_child in new_children.iter().clone() {
        for plan in new_child.data.to_linear_plans.clone() {
            let contains = linear_plans.iter().any(|p| Arc::ptr_eq(p, &plan));
            if !contains {
                linear_plans.push(plan);
            }
        }
    }
    // Capture the plan before replacing with new children so that we can compare the Arcs safely
    let plan_captured = Arc::clone(&new_node.plan);
    let mut res = new_node.with_new_children(new_children)? as SortPushDown;

    // println!("Linear plans plan is {:?}", plan_captured);
    // if !linear_plans.is_empty() {
    //     // println!("Not empty linear plans Soft requirements {}", res.data.soft_requirement_plans.len());
    //     // println!("Linear plan is {:?}", linear_plans[0]);
    // }
    // println!("Linear plans {}", linear_plans.len());
    for plan in linear_plans {
        // println!("Linear Plan is: {:?}", plan);
        if Arc::ptr_eq(&plan, &plan_captured) {
            // TODO Bunu API ile yap! downcast etme! Ve else'ine unreachable koyarak bak.
            if let Some(bounded) = plan.as_any().downcast_ref::<BoundedWindowAggExec>() {
                // println!("Changing...");
                res.plan = Arc::new(BoundedWindowAggExec::try_new(
                    bounded.window_expr().to_vec(),
                    // Use new child
                    Arc::clone(&res.children[0].plan),
                    InputOrderMode::Linear,
                    *bounded.can_repartition(),
                )?);
                // println!("Assign... {} {}",res.data.soft_requirement_plans.len(),res.data.to_linear_plans.len() );
                // If there are soft-required parents update them as well
                res.data.to_linear_plans = res.data.soft_requirement_plans.clone();
                res.data.soft_requirement_plans = vec![];
            } else {
                // println!("\n\nTo Linear plan is not BoundedWindowAgg {:?}\n\n", plan);
                res.data.to_linear_plans = res.data.soft_requirement_plans.clone();
                res.data.soft_requirement_plans = vec![];
            }
        } else {
            // println!("Plan is not equal");
            for child in &res.children {
                if !child.data.to_linear_plans.is_empty() {
                    res.data.to_linear_plans = child.data.to_linear_plans.clone();
                }
            }
        }
    }
    Ok(res)
}

fn min_fetch(f1: Option<usize>, f2: Option<usize>) -> Option<usize> {
    match (f1, f2) {
        (Some(f1), Some(f2)) => Some(f1.min(f2)),
        (Some(_), _) => f1,
        (_, Some(_)) => f2,
        _ => None,
    }
}

/// If there's a Hard Parent requirement returns Hard
/// Otherwise checks KeepHardRequirement flag and specification, if possible returns a Soft requirement
fn decide_sort_exec_requirement(
    output_ordering: &RequiredInputOrdering,
    parent_reqs: &RequiredInputOrdering,
    sort_push_down: &SortPushDown,
    current_is_stricter: bool,
) -> RequiredInputOrdering {
    // set the stricter ordering
    // println!("Decide SortExec Req, current is stricter {current_is_stricter}");
    let lex_requirement = if current_is_stricter {
        output_ordering.lex_requirement().clone()
    } else {
        parent_reqs.lex_requirement().clone()
    };

    if let Some(hard_parent) = sort_push_down
        .data
        .hard_parent_requirement
        .specification
        .clone()
    {
        hard_parent
    } else if parent_reqs.is_hard_and_non_empty() {
        RequiredInputOrdering::Hard(lex_requirement)
    } else {
        // There's no Hard parent requirement, decide based on context flag
        if sort_push_down.data.keep_hard_requirement.keep_hard {
            sort_push_down
                .data
                .keep_hard_requirement
                .check_based_on_specification(
                    Arc::clone(&sort_push_down.plan),
                    output_ordering.clone(),
                )
        } else {
            RequiredInputOrdering::Soft(lex_requirement)
        }
    }
}

/// If a SortExec can not satisfy its parent, this function is called.
/// It gets the parent sort_push_down and fetch info and adds a [`SortExec`] with them
fn add_sort_for_not_satisfied_sort_exec_parent(
    sort_push_down: SortPushDown,
    sort_fetch: Option<usize>,
) -> SortPushDown {
    let mut new_sort_push_down = sort_push_down;
    // Make sure this `SortExec` satisfies parent sort_push_down:
    let sort_reqs = new_sort_push_down
        .data
        .ordering_requirement
        .unwrap_or_default();
    // It's possible current plan (`SortExec`) has a fetch value.
    // And if both of them have fetch values, we should use the minimum one.
    if let Some(fetch) = sort_fetch {
        if let Some(requirement_fetch) = new_sort_push_down.data.fetch {
            new_sort_push_down.data.fetch = Some(fetch.min(requirement_fetch));
        }
    }
    let fetch = new_sort_push_down.data.fetch.or(sort_fetch);
    new_sort_push_down = new_sort_push_down.children.swap_remove(0);
    if sort_fetch.is_some() {
        new_sort_push_down.data.keep_hard_requirement.keep_hard = true;
        new_sort_push_down.data.keep_hard_requirement.specification = None;
    }
    // println!("Adding soft sort");
    new_sort_push_down = add_sort_above(
        new_sort_push_down,
        sort_reqs.mixed_lex_requirement().clone(),
        fetch,
    );
    new_sort_push_down
}

fn remove_soft_sort_exec_if_possible(
    mut sort_push_down: SortPushDown,
    parent_reqs: &RequiredInputOrdering,
    required_ordering: &RequiredInputOrdering,
    sort_fetch: Option<usize>,
) -> SortPushDown {
    // TODO Burada bir SortExec'in kaldırılması lazım yukarıdakini satisfy ediyor ve aynısını require ediyorsa
    // If there's no Hard input order Requirement from parent, Soft SortExec can be removable
    if !parent_reqs.is_hard_and_non_empty()
        && !required_ordering.is_hard_and_non_empty()
        && sort_fetch.is_none()
        && sort_push_down
            .data
            .hard_parent_requirement
            .specification
            .is_none()
    // || (required_ordering == parent_reqs)
    {
        let old_keep_children_as_hard =
            sort_push_down.data.keep_hard_requirement.keep_hard;
        // println!("Removing Soft Sort Exec... {}", sort_push_down.data.soft_requirement_plans.len());
        let old_soft_plans = sort_push_down.data.soft_requirement_plans.clone();
        sort_push_down = sort_push_down.children.swap_remove(0);
        // Use inherited value or set child's fetch information
        sort_push_down.data.keep_hard_requirement.keep_hard =
            old_keep_children_as_hard || sort_push_down.plan.fetch().is_some();
        if sort_push_down.plan.fetch().is_some() {
            sort_push_down.data.keep_hard_requirement.specification = None;
        }
        sort_push_down.data.to_linear_plans = old_soft_plans.clone();
        sort_push_down.data.soft_requirement_plans = old_soft_plans;
    } else if parent_reqs.is_hard_and_non_empty() {
        match parent_reqs {
            RequiredInputOrdering::Mixed((hard, mixed)) => {
                // TODO Burada Soft'a göre değiştirdik ama yukarıdakileri değiştirebilir seçmek lazım.
                // TODO Soft için deneyip yukarıdan aşağı gelen planları tutup, onlar için kuralı tekrar çağırsak? Sort sayısına göre karar versek?

                // TODO Burada max planı satisfy edecek sort permütasyonunu seçmemiz lazım. Visited plans tut (Mixed plans'ı tutarak karar verelim demiştik ama planlar değişiyor)

                // TODO Keep hard sort_push_down'ı kullanabilir miyiz? Tek bir requirement'ı sanki saklayıp onu satisfy ediyorsa bunu yap demek lazım.
                // println!("Parent mixed, changing sort To Linear Count: {} Soft count: {} Keep Hard {:?}\n Hard Parent {:?}", sort_push_down.data.to_linear_plans.len(), sort_push_down.data.soft_requirement_plans.len(), sort_push_down.data.keep_hard_requirement.clone(), sort_push_down.data.hard_parent_requirement);
                // We already have the SortExec, replace it with soft sort_push_down
                // because we couldn't avoid Sorting operation anyway
                if sort_push_down
                    .data
                    .hard_parent_requirement
                    .specification
                    .is_some()
                {
                    if let Some(spec) = sort_push_down
                        .data
                        .hard_parent_requirement
                        .specification
                        .clone()
                    {
                        if !sort_push_down
                            .plan
                            .equivalence_properties()
                            .ordering_satisfy_requirement(spec.lex_requirement())
                        {
                            let mut new_inner = vec![];
                            for sort_req in &spec.lex_requirement().inner {
                                if let Some(col) =
                                    sort_req.expr.as_any().downcast_ref::<Column>()
                                {
                                    let current_schema = sort_push_down.plan.schema();
                                    let (i, column) = current_schema
                                        .column_with_name(col.name())
                                        .unwrap();
                                    new_inner.push(PhysicalSortRequirement::new(
                                        Arc::new(Column::new(column.name(), i)),
                                        sort_req.options,
                                    ));
                                }
                            }
                            let spec = RequiredInputOrdering::Hard(LexRequirement::new(
                                new_inner,
                            ));
                            // TODO Burada Spec'e güncelliyoruz ama Child requirement ile çelişiyorsa başka bir şey yapmalıyız. Karar vermeli.
                            // println!("Not satisfying... change {:?}\n\nSoft {}", spec, sort_push_down.data.soft_requirement_plans.len());
                            let old_soft_plans =
                                sort_push_down.data.soft_requirement_plans;
                            sort_push_down = sort_push_down.children.swap_remove(0);
                            sort_push_down = add_sort_above(
                                sort_push_down,
                                spec.lex_requirement().clone(),
                                sort_fetch,
                            );
                            // println!("Not satisfying 2... Soft {}",sort_push_down.data.soft_requirement_plans.len() );
                            // TODO 1 tane çıkarmak çözmez. Kaç tane varsa o kadar çıkarmak lazım
                            sort_push_down.data.to_linear_plans = old_soft_plans;
                            // println!("Not satisfying 3... To linear {}",sort_push_down.data.to_linear_plans.len());
                            // if let Some(soft_plan) = sort_push_down.data.soft_requirement_plans.pop() {
                            //     // println!("Soft plan...");
                            //     if !soft_plan.equivalence_properties().ordering_satisfy_requirement(spec.lex_requirement()) {
                            //         // println!("Re-add to soft plan 1");
                            //         sort_push_down.data.soft_requirement_plans.push(soft_plan);
                            //     }
                            // }
                        } else {
                            // If Current Sort is satisfying the higher Hard Requirement but missing the current mixed requirement change it to linear

                            // TODO Hard Requirement'ın Hard'ını satisfy ediyor ama Mixed'ini etmiyorsa Sort'u mixed'e çevir
                            // TODO 2 Bunu sadece Mixed hali hard'ı satisfy ediyorsa yapabiliriz.

                            // Plan is BoundedWindowAggExec { input: SortExec { input: BoundedWindowAggExec { input: SortExec { input: BoundedWindowAggExec { input: SortExec { input: BoundedWindowAggExec { input: ProjectionExec { expr: [(CastExpr { expr: Column { name: "c", index: 2 }, cast_type: Int64, cast_options: CastOptions { safe: false, format_options: FormatOptions { safe: true, null: "", date_

                            if spec.lex_requirement().clone() != mixed.clone() {
                                // println!("Hard Requirement is not equal to parent mixed");
                                sort_push_down.data.to_linear_plans =
                                    sort_push_down.data.soft_requirement_plans.clone();
                            } else {
                                // println!("Hard Requirement IS equal to parent mixed... no change");
                            }
                        }
                    } else {
                        // println!("No specification!");
                        let old_soft_plans = sort_push_down.data.soft_requirement_plans;
                        sort_push_down = sort_push_down.children.swap_remove(0);
                        sort_push_down =
                            add_sort_above(sort_push_down, mixed.clone(), sort_fetch);
                        sort_push_down.data.soft_requirement_plans = old_soft_plans;
                        // TODO 1 tane çıkarmak çözmez. Kaç tane varsa o kadar çıkarmak lazım
                        if let Some(soft_plan) =
                            sort_push_down.data.soft_requirement_plans.pop()
                        {
                            if !soft_plan
                                .equivalence_properties()
                                .ordering_satisfy_requirement(mixed)
                            {
                                // println!("Re-add to soft plan 2");
                                sort_push_down
                                    .data
                                    .soft_requirement_plans
                                    .push(soft_plan);
                            }
                        }
                    }
                } else {
                    // println!("No Hard Parent! Mixed {:?}", mixed.clone());
                    let old_soft_plans = sort_push_down.data.soft_requirement_plans;
                    sort_push_down = sort_push_down.children.swap_remove(0);
                    // TODO Burada eğer Sort eklemekten kaçınabilirsek, kaçınmamız lazım.
                    if sort_push_down
                        .plan
                        .equivalence_properties()
                        .ordering_satisfy_requirement(mixed)
                        || sort_push_down
                            .plan
                            .equivalence_properties()
                            .ordering_satisfy_requirement(hard)
                    {
                        // println!("Satisfies mixed or hard");
                    } else {
                        // println!("Does not satisfy mixed or hard");
                        let requirements = sort_push_down.plan.required_input_ordering();
                        for maybe_req in requirements {
                            if let Some(req) = maybe_req {
                                let mut hard_copy = hard.clone();
                                while hard_copy.inner.len() == 0 {
                                    if sort_push_down
                                        .plan
                                        .equivalence_properties()
                                        .requirements_compatible(
                                            &hard_copy,
                                            req.lex_requirement(),
                                        )
                                    {
                                        // println!("Hard is compatible!");
                                    }
                                    hard_copy = LexRequirement {
                                        inner: hard_copy.inner
                                            [0..hard_copy.inner.len() - 1]
                                            .to_vec(),
                                    }
                                }
                                let mut mixed_copy = mixed.clone();
                                while mixed_copy.inner.len() == 0 {
                                    if sort_push_down
                                        .plan
                                        .equivalence_properties()
                                        .requirements_compatible(
                                            &mixed_copy,
                                            req.lex_requirement(),
                                        )
                                    {
                                        // println!("Mixed is compatible!");
                                    }
                                    hard_copy = LexRequirement {
                                        inner: mixed_copy.inner
                                            [0..mixed_copy.inner.len() - 1]
                                            .to_vec(),
                                    }
                                }
                            }
                        }
                        sort_push_down =
                            add_sort_above(sort_push_down, mixed.clone(), sort_fetch);
                    }
                    sort_push_down.data.soft_requirement_plans = old_soft_plans;
                    // TODO 1 tane çıkarmak çözmez. Kaç tane varsa o kadar çıkarmak lazım
                    if let Some(soft_plan) =
                        sort_push_down.data.soft_requirement_plans.pop()
                    {
                        // println!("Soft plan exists");
                        if !soft_plan
                            .equivalence_properties()
                            .ordering_satisfy_requirement(mixed)
                        {
                            // println!("Re-add to soft plan 3");
                            // sort_push_down.data.soft_requirement_plans.push(soft_plan);
                        }
                    }
                }
                // println!("Parent mixed, changing sort To Linear Count: {} Soft count: {}", sort_push_down.data.to_linear_plans.len(),sort_push_down.data.soft_requirement_plans.len() );
            }
            _ => {}
        }
        sort_push_down.data.keep_hard_requirement.keep_hard = true;
        sort_push_down.data.keep_hard_requirement.specification =
            Some(parent_reqs.clone());
    } else {
        sort_push_down.data.keep_hard_requirement.keep_hard = false;
    }
    sort_push_down
}

fn assign_requirements_to_satisfying_child(
    child: &mut PlanContext<ParentRequirements>,
    parent_reqs: &RequiredInputOrdering,
    child_req: &RequiredInputOrdering,
    plan: &Arc<dyn ExecutionPlan>,
    keep_hard_requirement: &mut KeepHardRequirement,
    hard_parent_requirement: &mut KeepHardRequirement,
) -> RequiredInputOrdering {
    let contains_in_soft_plans = child
        .data
        .soft_requirement_plans
        .iter()
        .any(|p| Arc::ptr_eq(p, plan));
    match (parent_reqs.clone(), child_req.clone()) {
        (_, RequiredInputOrdering::Mixed(mixed)) => {
            child.data.keep_hard_requirement = keep_hard_requirement.clone();
            // println!("Child requirements are mixed");
            if keep_hard_requirement.keep_hard {
                let Some(spec) = keep_hard_requirement.specification.clone() else {
                    // println!("Has Limit, turning Mixed into Hard");
                    return RequiredInputOrdering::Hard(mixed.1);
                };
                // TODO Mixed ile eşleyorsa
                if spec.lex_requirement().clone() == mixed.1 {
                    // println!("Specs matched, turning Mixed into Hard");
                    return RequiredInputOrdering::Hard(mixed.1);
                }
                // println!("Returning mixed...");

                if !contains_in_soft_plans {
                    child.data.soft_requirement_plans.push(Arc::clone(plan));
                }
                RequiredInputOrdering::Mixed(mixed)
            } else {
                // println!("Keep hard false...");
                if !contains_in_soft_plans {
                    child.data.soft_requirement_plans.push(Arc::clone(plan));
                }
                RequiredInputOrdering::Mixed(mixed)
            }
            // assign_requirements_to_satisfying_child(child, parent_reqs, &RequiredInputOrdering::Hard(mixed), plan, keep_hard_requirement)
        }
        // TODO Mixed'i düzgün handle et, sadece Hard'a cast ettik
        (RequiredInputOrdering::Mixed(mixed), _) => {
            // println!("Parent requirements are mixed");
            RequiredInputOrdering::Mixed(mixed)
        }
        // There's no parent requirement, if child is also not Soft, Hard requirement can be returned
        (RequiredInputOrdering::Hard(lex), RequiredInputOrdering::Hard(_))
            if lex.is_empty() =>
        {
            // println!( "Parent requirements are empty, but child is hard assigning hard {:?}", child_req.lex_requirement().clone() );
            keep_hard_requirement.keep_hard = true;
            keep_hard_requirement.specification = Some(child_req.clone());
            hard_parent_requirement.keep_hard = true;
            hard_parent_requirement.specification = Some(child_req.clone());

            RequiredInputOrdering::Hard(child_req.lex_requirement().clone())
        }
        // There's no parent requirement, if child is Soft, requirement is Soft, and add it to soft_requirement_plans
        (RequiredInputOrdering::Hard(lex), RequiredInputOrdering::Soft(_))
            if lex.is_empty() =>
        {
            // println!( "Parent requirements are empty, child is soft {:?}", child_req.to_vec() );
            if keep_hard_requirement.keep_hard {
                let res = keep_hard_requirement
                    .check_based_on_specification(Arc::clone(plan), child_req.clone());
                if matches!(res, RequiredInputOrdering::Soft(_))
                    && !contains_in_soft_plans
                {
                    child.data.soft_requirement_plans.push(Arc::clone(plan));
                }
                res
            } else {
                if !contains_in_soft_plans {
                    child.data.soft_requirement_plans.push(Arc::clone(plan));
                }
                RequiredInputOrdering::Soft(LexRequirement::new(child_req.to_vec()))
            }
        }
        // There's a parent or child hard requirement
        (RequiredInputOrdering::Hard(_), _)
        | (RequiredInputOrdering::Soft(_), RequiredInputOrdering::Hard(_)) => {
            // println!("Parent or child has hard requirement Parent: {:?}\nChild: {:?}", parent_reqs, child_req );
            // TODO Burada parent'ı alsak joins.slt çocuğu alsak window.slt patlıyor. Galiba bu çocuk mu strict parent mı bug'ının çözümünden sonra doğru olacak.

            // TODO Mixed'den gelen Hard parent'ı satisfy ediyor ama Sorted mode için partition by indices zorunlu tutuluyor. Öyleyse Sorted mode olmamalı mı?
            child.data.keep_hard_requirement = keep_hard_requirement.clone();
            let child_is_strict =
                child.plan.equivalence_properties().requirements_compatible(
                    child_req.lex_requirement(),
                    parent_reqs.lex_requirement(),
                );
            if child_is_strict {
                // println!("Child is strict ");
                RequiredInputOrdering::Hard(LexRequirement::new(child_req.to_vec()))
            } else {
                // println!("Child is not strict!");
                if plan
                    .equivalence_properties()
                    .ordering_satisfy_requirement(parent_reqs.lex_requirement())
                {
                    // println!("Child is not strict satisfy!");
                    RequiredInputOrdering::Hard(LexRequirement::new(child_req.to_vec()))
                } else {
                    // println!("Child is not strict no satisfy!");
                    RequiredInputOrdering::Hard(LexRequirement::new(parent_reqs.to_vec()))
                }
            }
        }
        // Both requirements are soft
        (RequiredInputOrdering::Soft(_), RequiredInputOrdering::Soft(_)) => {
            if keep_hard_requirement.keep_hard {
                keep_hard_requirement
                    .check_based_on_specification(Arc::clone(plan), child_req.clone())
            } else {
                if !contains_in_soft_plans {
                    child.data.soft_requirement_plans.push(Arc::clone(plan));
                }
                RequiredInputOrdering::Soft(LexRequirement::new(child_req.to_vec()))
            }
        }
    }
}

fn assign_requirement_to_all_children(
    sort_push_down: &mut SortPushDown,
    soft_plans: Vec<Arc<dyn ExecutionPlan>>,
    reqs: Vec<Option<RequiredInputOrdering>>,
) {
    let mut soft_plans = soft_plans.clone();
    for (child, requirement) in sort_push_down.children.iter_mut().zip(reqs) {
        child.data = ParentRequirements {
            ordering_requirement: requirement,
            fetch: child.plan.fetch(),
            soft_requirement_plans: soft_plans.clone(),
            // TODO soft plans yerine to linear plans'ı atadık
            to_linear_plans: sort_push_down.data.to_linear_plans.clone(),
            keep_hard_requirement: sort_push_down.data.keep_hard_requirement.clone(),
            hard_parent_requirement: sort_push_down.data.hard_parent_requirement.clone(),
        };
    }
}

/// Assigns a SortExec's requirements to its child and grand children with state since it has been removed.
fn assign_pushed_down_requirement_to_sort_children(
    child: &mut PlanContext<ParentRequirements>,
    sort_push_down: SortPushDown,
    adjusted: Vec<Option<RequiredInputOrdering>>,
    required_ordering: RequiredInputOrdering,
    soft_plans: Vec<Arc<dyn ExecutionPlan>>,
    fetch: Option<usize>,
) {
    let mut cloned_soft_plans = soft_plans.clone();
    let mut to_linear_plans = sort_push_down.data.to_linear_plans.clone();
    to_linear_plans.append(&mut cloned_soft_plans);
    for (grand_child, order) in child.children.iter_mut().zip(adjusted) {
        grand_child.data = ParentRequirements {
            ordering_requirement: order,
            fetch,
            soft_requirement_plans: soft_plans.clone(),
            to_linear_plans: to_linear_plans.clone(),
            keep_hard_requirement: sort_push_down.data.keep_hard_requirement.clone(),
            hard_parent_requirement: sort_push_down.data.hard_parent_requirement.clone(),
        };
    }
    child.data = ParentRequirements {
        ordering_requirement: Some(required_ordering.clone()),
        fetch,
        soft_requirement_plans: soft_plans,
        to_linear_plans: to_linear_plans.clone(),
        keep_hard_requirement: sort_push_down.data.keep_hard_requirement.clone(),
        hard_parent_requirement: sort_push_down.data.hard_parent_requirement.clone(),
    };
}

/// The main optimizer helper for [`pushdown_sorts`]. Holds a state with [`SortPushDown`] type.
///
/// # Context
///
/// While pushing down input ordering requirements uses this state to be able to use the parents' sort_push_down.
/// Firstly, checks if the current plan is a [`SortExec`]. If so, identifies if this sort is soft/hard requirement
/// based on parent requirements and fetch information either removes the SortExec, modifies with fetch info or keeps as is
///
/// If the plan is not a [`SortExec`] checks if it satisfies parent requirements, if so
/// decides the [`RequiredInputOrdering`] based on current plans requirement and parent, and sets into the children state
///
/// If both is not true, tries to push down the parent requirements below the plan, if it can
/// it sets the current requirement into the children state
///
/// If could not do anything, based on parent requirements checks if a new [`SortExec`] is necessary, if so adds one
fn pushdown_sorts_helper(
    mut sort_push_down: SortPushDown,
) -> Result<Transformed<SortPushDown>> {
    let parent_reqs = sort_push_down
        .data
        .ordering_requirement
        .clone()
        .unwrap_or_default();
    let soft_plans = sort_push_down.data.soft_requirement_plans.clone();
    let satisfy_parent = sort_push_down
        .plan
        .equivalence_properties()
        .ordering_satisfy_requirement(parent_reqs.lex_requirement());
    if sort_push_down.plan.fetch().is_some() {
        // println!("Fetch is there, no specification...");
        sort_push_down.data.keep_hard_requirement.keep_hard = true;
        sort_push_down.data.keep_hard_requirement.specification = None;
    }
    // println!("\nPlan is {:?}\n\n To Linear count {} Soft count {} Satisfy parent? {satisfy_parent} Parent Requirements {:?}\n Keep Hard Requirements {:?}\n Hard Parent Requirements {:?}", sort_push_down.plan, sort_push_down.data.to_linear_plans.len(), sort_push_down.data.soft_requirement_plans.len(), parent_reqs, sort_push_down.data.keep_hard_requirement, sort_push_down.data.hard_parent_requirement);

    if is_sort(&sort_push_down.plan) {
        let current_sort_fetch = sort_push_down.plan.fetch();
        let parent_req_fetch = sort_push_down.data.fetch;
        let output_ordering = &sort_push_down
            .plan
            .output_ordering()
            .cloned()
            .map(LexRequirement::from)
            .map(RequiredInputOrdering::Hard)
            .unwrap_or_default();
        let current_is_stricter = sort_push_down
            .plan
            .equivalence_properties()
            .requirements_compatible(
                output_ordering.lex_requirement(),
                parent_reqs.lex_requirement(),
            );
        // println!("Is sort! Fetch {:?} Output ordering {:?} ", current_sort_fetch, output_ordering);

        // Decide SortExec's required ordering based on ParentRequirement or previous keep_hard_requirements information
        let required_ordering = decide_sort_exec_requirement(
            output_ordering,
            &parent_reqs,
            &sort_push_down,
            current_is_stricter,
        );
        if !satisfy_parent && parent_reqs.is_hard_and_non_empty() {
            // if !satisfy_parent && !parent_is_stricter && parent_reqs.is_hard_and_non_empty() {
            // println!("Adding sort for not satisfied SortExec");
            // TODO Burada Soft'a göre ekledik ama yukarıdakileri değiştirebilir seçmek lazım.
            // TODO Soft için deneyip yukarıdan aşağı gelen planları tutup, onlar için kuralı tekrar çağırsak? Sort sayısına göre karar versek?
            sort_push_down = add_sort_for_not_satisfied_sort_exec_parent(
                sort_push_down,
                current_sort_fetch,
            )
        };

        // We can safely get the 0th index as we are dealing with a `SortExec`.
        let child = sort_push_down.children.first().unwrap();
        // set the stricter fetch
        let pushdown_result = pushdown_requirement_to_children(
            Arc::clone(&child.plan),
            &required_ordering,
            soft_plans.clone(),
        )?;

        // println!("Sort pushed down? {:?} Required Ordering {:?}", pushdown_result.required_input_ordering, required_ordering);
        if let Some(adjusted) = pushdown_result.required_input_ordering {
            // Push down is successful, remove the SortExec and set it as ordering requirement
            let mut child = sort_push_down.children.swap_remove(0);
            // TODO Pushdown etmişsin, Sort'u sildin hard requirement'ı aşağı verip yukarıdakini Linear'e çevir.

            assign_pushed_down_requirement_to_sort_children(
                &mut child,
                sort_push_down,
                adjusted,
                required_ordering,
                pushdown_result.soft_requirement_plans,
                min_fetch(current_sort_fetch, parent_req_fetch),
            );

            // println!("\nReturn child {:?}\n", child.plan);
            return Ok(Transformed {
                data: child,
                transformed: true,
                tnr: TreeNodeRecursion::Stop,
            });
        }

        // Can not push down requirements
        sort_push_down = remove_soft_sort_exec_if_possible(
            sort_push_down,
            &parent_reqs,
            &required_ordering,
            current_sort_fetch,
        );
        if let Some(hard_req) = &sort_push_down.data.hard_parent_requirement.specification
        {
            if sort_push_down
                .plan
                .equivalence_properties()
                .ordering_satisfy_requirement(hard_req.lex_requirement())
            {
                // println!("Sort exec is satisfying hard requirement...");
                sort_push_down.data.hard_parent_requirement.specification = None;
                sort_push_down.data.hard_parent_requirement.keep_hard = false;

                sort_push_down.data.keep_hard_requirement.keep_hard = false;
                sort_push_down.data.keep_hard_requirement.specification = None;

                // sort_push_down.data.soft_requirement_plans = vec![];
                // println!("To Linear plans are {:?}", sort_push_down.data.to_linear_plans);
            }
        }
        let soft_plans = sort_push_down.data.soft_requirement_plans.clone();

        let reqs = sort_push_down.plan.required_input_ordering();
        // Assign input order requirement information to children data
        assign_requirement_to_all_children(&mut sort_push_down, soft_plans, reqs);
        // println!("Could not push down sort, requirements assigned to child S: {} L: {}",sort_push_down.data.soft_requirement_plans.len(),sort_push_down.data.to_linear_plans.len() );
        // } else if parent_reqs.is_empty() {
        //     // println!("Parent requirements are empty!");
        //     // note: this `satisfy_parent`, but we don't want to push down anything.
        //     // Nothing to do.
        //     return Ok(Transformed::no(sort_push_down));
    } else if satisfy_parent {
        // For non-sort operators, parent requirements are met:
        let reqs = &sort_push_down.plan.required_input_ordering();
        // println!("Satisfies parent... plans required ordering {:?}", reqs);
        // TODO Burada permütasyonları da kontrol etmemiz lazım
        if matches!(parent_reqs, RequiredInputOrdering::Mixed(_))
            && sort_push_down.children.is_empty()
        {
            // println!("Parent req add sort");
            if !sort_push_down
                .plan
                .equivalence_properties()
                .ordering_satisfy_requirement(parent_reqs.mixed_lex_requirement())
            {
                // println!("To linear...");
                sort_push_down.data.to_linear_plans =
                    sort_push_down.data.soft_requirement_plans.clone();
            }
        } else if matches!(
            sort_push_down.data.keep_hard_requirement.specification,
            Some(RequiredInputOrdering::Mixed(_))
        ) && sort_push_down.children.is_empty()
        {
            // println!("Keep Hard req add sort");
            if !sort_push_down
                .plan
                .equivalence_properties()
                .ordering_satisfy_requirement(
                    sort_push_down
                        .data
                        .keep_hard_requirement
                        .specification
                        .clone()
                        .unwrap()
                        .mixed_lex_requirement(),
                )
            {
                // println!("To linear...");
                sort_push_down.data.to_linear_plans =
                    sort_push_down.data.soft_requirement_plans.clone();
            }
        } else if matches!(
            sort_push_down.data.hard_parent_requirement.specification,
            Some(RequiredInputOrdering::Mixed(_))
        ) && sort_push_down.children.is_empty()
        {
            // println!("Hard Parent req add sort");
            if !sort_push_down
                .plan
                .equivalence_properties()
                .ordering_satisfy_requirement(
                    sort_push_down
                        .data
                        .hard_parent_requirement
                        .specification
                        .clone()
                        .unwrap()
                        .mixed_lex_requirement(),
                )
            {
                // println!("To linear...");
                sort_push_down.data.to_linear_plans =
                    sort_push_down.data.soft_requirement_plans.clone();
            }
        }

        for (child, order) in sort_push_down.children.iter_mut().zip(reqs) {
            let Some(child_req) = order else {
                // println!("Child does not require ordering...");
                // There are no child requirements, so we will remove the parent requirements from ordering_requirement,
                // but we'll still keep the `keep_children_as_hard` information since upper fetch operators or a Hard parent may affect it
                child.data.ordering_requirement = None;
                if parent_reqs.is_hard_and_non_empty() {
                    // println!("Setting keep hard requirement");
                    child.data.keep_hard_requirement.specification =
                        Some(parent_reqs.clone());
                    child.data.keep_hard_requirement.keep_hard = true;
                } else {
                    // println!("Inheriting keep hard requirement");
                    child.data.keep_hard_requirement =
                        sort_push_down.data.keep_hard_requirement.clone();
                }
                child.data.soft_requirement_plans =
                    sort_push_down.data.soft_requirement_plans.clone();
                child.data.to_linear_plans = sort_push_down.data.to_linear_plans.clone();
                child.data.hard_parent_requirement =
                    sort_push_down.data.hard_parent_requirement.clone();
                continue;
            };
            let order = assign_requirements_to_satisfying_child(
                child,
                &parent_reqs,
                child_req,
                &sort_push_down.plan,
                &mut sort_push_down.data.keep_hard_requirement,
                &mut sort_push_down.data.hard_parent_requirement,
            );
            child.data.ordering_requirement = Some(order);
            // TODO bunu yukarıda bir yerde çağırıp karar verip her yerde kullanalım
            let child_is_strict =
                child.plan.equivalence_properties().requirements_compatible(
                    child_req.lex_requirement(),
                    parent_reqs.lex_requirement(),
                );
            if child_is_strict
                && sort_push_down
                    .data
                    .keep_hard_requirement
                    .specification
                    .is_some()
            {
                child.data.hard_parent_requirement.specification =
                    Some(child_req.clone());
            } else {
                child.data.hard_parent_requirement =
                    sort_push_down.data.hard_parent_requirement.clone();
            }
            // println!("Requirements assigned to child L: {:?} S: {}", child.data.to_linear_plans.len(), child.data.soft_requirement_plans.len());
            sort_push_down.data.to_linear_plans = child.data.to_linear_plans.clone();
            let mut new_soft_plans = child.data.soft_requirement_plans.clone();
            let mut new_linear_plans = child.data.to_linear_plans.clone();
            sort_push_down
                .data
                .soft_requirement_plans
                .append(&mut new_soft_plans);
            sort_push_down
                .data
                .to_linear_plans
                .append(&mut new_linear_plans);
            child.data.soft_requirement_plans =
                sort_push_down.data.soft_requirement_plans.clone();
            child.data.to_linear_plans = sort_push_down.data.to_linear_plans.clone();
            // println!("Requirements lengths soft: {} lin: {}", sort_push_down.data.soft_requirement_plans.len(),sort_push_down.data.to_linear_plans.len() );
        }
    } else {
        // Can not satisfy the parent requirements, check whether we can push requirements down:
        let pushdown_result = pushdown_requirement_to_children(
            Arc::clone(&sort_push_down.plan),
            &parent_reqs,
            soft_plans.clone(),
        )?;
        // println!("Non-satisfying child push down result {:?}", pushdown_result.required_input_ordering);
        if let Some(adjusted) = pushdown_result.required_input_ordering {
            if let Some(add_sort_req) = pushdown_result.add_sort_requirement {
                // println!("Non-satisfying child add sort {:?}", add_sort_req);
                let fetch = sort_push_down.data.fetch;
                sort_push_down = add_sort_above(
                    sort_push_down,
                    add_sort_req.mixed_lex_requirement().clone(),
                    fetch,
                );
                sort_push_down.data.hard_parent_requirement =
                    KeepHardRequirement::default();
                sort_push_down.data.soft_requirement_plans = vec![];
            }
            for (child, order) in sort_push_down.children.iter_mut().zip(adjusted) {
                child.data.ordering_requirement = order.clone();
                child.data.keep_hard_requirement.keep_hard =
                    sort_push_down.data.keep_hard_requirement.keep_hard;
                if sort_push_down
                    .data
                    .hard_parent_requirement
                    .specification
                    .is_some()
                {
                    child.data.hard_parent_requirement.specification = order;
                }
            }
            sort_push_down.data.ordering_requirement = None;
            sort_push_down.data.to_linear_plans =
                pushdown_result.soft_requirement_plans.clone();
            sort_push_down.data.soft_requirement_plans =
                pushdown_result.soft_requirement_plans.clone();
            sort_push_down.data.soft_requirement_plans =
                pushdown_result.soft_requirement_plans.clone();
            // println!("Non-satisfying child pushed down requirements S: {} L: {}", sort_push_down.data.soft_requirement_plans.len(), sort_push_down.data.to_linear_plans.len());
        } else {
            // Can not push down requirements, check if can add a new `SortExec`
            let sort_reqs = sort_push_down
                .data
                .ordering_requirement
                .clone()
                .unwrap_or_default();
            let required_fetch = sort_push_down.data.fetch;
            // println!("Non-satisfying child could not pushed down sort reqs {:?}, fetch {:?}",sort_reqs, required_fetch);
            // TODO Fetch varsa da yapmak gerekmez mi?
            if let Some(spec) = sort_push_down
                .data
                .hard_parent_requirement
                .specification
                .clone()
            {
                if !sort_push_down
                    .plan
                    .equivalence_properties()
                    .ordering_satisfy_requirement(spec.mixed_lex_requirement())
                {
                    // println!("Adding SortExec above... {:?}", spec.mixed_lex_requirement());
                    sort_push_down = add_sort_above(
                        sort_push_down,
                        spec.mixed_lex_requirement().clone(),
                        required_fetch,
                    );
                    sort_push_down.data.hard_parent_requirement =
                        KeepHardRequirement::default();
                    // TODO Burada SoftPlans'tan 1 tane çıkarmalı mı? 14.03.2025
                    sort_push_down.data.to_linear_plans = soft_plans;
                    sort_push_down.data.soft_requirement_plans = vec![];
                    // println!("SortExec added above. L: {} S: {}", sort_push_down.data.to_linear_plans.len(), sort_push_down.data.soft_requirement_plans.len());
                }
                // TODO Eğer zaaten satisfy ediliyorsa eklemeyebiliriz...
            } else if sort_reqs.is_hard_and_non_empty() {
                // println!("Adding SortExec above 2... {:?}", sort_reqs.mixed_lex_requirement());
                sort_push_down = add_sort_above(
                    sort_push_down,
                    sort_reqs.mixed_lex_requirement().clone(),
                    required_fetch,
                );
                // println!("SortExec added due to requirements")
            }
            let reqs = sort_push_down.plan.required_input_ordering();
            // TODO Burada eğer Soft requirement varsa eklememiz lazım
            let soft_plans = sort_push_down.data.soft_requirement_plans.clone();
            assign_requirement_to_all_children(&mut sort_push_down, soft_plans, reqs);
        }
    }

    Ok(Transformed::yes(sort_push_down))
}

/// When the required input ordering is satisfied decide child's required input ordering
/// If there's no parent requirement, consider child's requirement as soft.
/// If the parent requirement is Hard, keep it.
/// If the parent requirement is soft but child requirement is hard, set as hard requirement.
/// If both can be considered as soft requirements, set child as soft.
fn determine_satisfied_requirement_for_window(
    parent_required: RequiredInputOrdering,
    request_child: RequiredInputOrdering,
    plan: Arc<dyn ExecutionPlan>,
    soft_plans: &mut Vec<Arc<dyn ExecutionPlan>>,
) -> RequiredInputOrdering {
    let contains = soft_plans.iter().any(|p| Arc::ptr_eq(p, &plan));
    match (parent_required, request_child.clone()) {
        (RequiredInputOrdering::Hard(parent_req), _) => {
            // println!("Hard Parent...");
            if !parent_req.is_empty() {
                RequiredInputOrdering::Hard(LexRequirement::new(request_child.to_vec()))
            } else {
                if !contains {
                    soft_plans.push(plan);
                }
                RequiredInputOrdering::Soft(request_child.lex_requirement().clone())
            }
        }
        (RequiredInputOrdering::Soft(_), RequiredInputOrdering::Hard(_)) => {
            // println!("Hard Child...");
            RequiredInputOrdering::Hard(LexRequirement::new(request_child.to_vec()))
        }
        (RequiredInputOrdering::Soft(_), RequiredInputOrdering::Soft(_)) => {
            if !contains {
                soft_plans.push(plan);
            }
            RequiredInputOrdering::Soft(LexRequirement::new(request_child.to_vec()))
        }
        (_, RequiredInputOrdering::Mixed((hard, mixed))) => {
            // println!("Mixed child");
            // determine_satisfied_requirement_for_window(
            //     RequiredInputOrdering::Hard(mixed),
            //     request_child,
            //     plan,
            //     soft_plans,
            // )
            // if !hard.is_empty() {
            //     return determine_satisfied_requirement_for_window(RequiredInputOrdering::Hard(hard), request_child, plan, soft_plans)
            // };
            // RequiredInputOrdering::Soft(LexRequirement::new(request_child.to_vec()))
            RequiredInputOrdering::Mixed((hard, mixed))
        }
        (RequiredInputOrdering::Mixed((hard, mixed)), _) => {
            // println!("Mixed parent");
            // TODO
            // determine_satisfied_requirement_for_window(
            //     RequiredInputOrdering::Hard(mixed),
            //     request_child,
            //     plan,
            //     soft_plans,
            // )
            RequiredInputOrdering::Mixed((hard, mixed))
        }
    }
}

/// Sort Pushdown attempt's result information.
struct PushdownRequirementToChildrenResult {
    /// Set to None if push down is not successful, otherwise returns the children's required input ordering
    required_input_ordering: Option<Vec<Option<RequiredInputOrdering>>>,
    /// Pushdown may modify given soft_requirement_plans, and returns the modified values
    soft_requirement_plans: Vec<Arc<dyn ExecutionPlan>>,
    /// Requirement that needs to be added as sort above
    add_sort_requirement: Option<RequiredInputOrdering>,
}

impl PushdownRequirementToChildrenResult {
    fn new(
        required_input_ordering: Option<Vec<Option<RequiredInputOrdering>>>,
        soft_requirement_plans: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Self {
        Self {
            required_input_ordering,
            soft_requirement_plans,
            add_sort_requirement: None,
        }
    }

    fn with_add_sort_requirement(mut self, req: RequiredInputOrdering) -> Self {
        self.add_sort_requirement = Some(req);
        self
    }
}

/// Tries to pushdown given plan's input order requirement into lower plans.
///
/// # Parameters
/// - Plan: The execution plan that has required input ordering
/// - Parent Required: State information for [`pushdown_sorts`], the requirement that is coming from parent plans
/// - Soft Requirement Plans: The plans that caused a requirement being [`RequiredInputOrdering::Soft`]
///     keeping this information to be able to change their Input Order Modes to Linear
///     if the related [`SortExec`] is disappeared during optimization
///
/// # Context
/// Checks the plan types and their attributes such as
/// - Plan has limit information and can push it down
/// - Plan is a leaf node or can not maintain its input orders
///
/// Based on this information decides about how to decide the new input ordering requirement,
///   and it's type ([`RequiredInputOrdering::Hard`] or [`RequiredInputOrdering::Soft`])
///
/// For example, if the plan is a window, checks whether it's child satisfies the requirements or not
/// (see [`determine_children_requirement`] for details)
/// and then decides ordering requirement and soft required plans.
///
/// # Return
/// Returns the final input order requirement as the first parameter
/// And as second parameters if any plans that caused to return a Soft plan, returns them
fn pushdown_requirement_to_children(
    plan: Arc<dyn ExecutionPlan>,
    parent_requirement: &RequiredInputOrdering,
    soft_requirement_plans: Vec<Arc<dyn ExecutionPlan>>,
) -> Result<PushdownRequirementToChildrenResult> {
    let maintains_input_order = plan.maintains_input_order();
    let mut soft_plans = soft_requirement_plans.clone();
    if is_window(&plan) {
        // println!("Is window!");
        let required_input_ordering = plan.required_input_ordering();
        let child_requirement = required_input_ordering[0].clone().unwrap_or_default();
        let child_plan = plan.children().swap_remove(0);

        let mut result = match determine_children_requirement(
            parent_requirement,
            &child_requirement,
            child_plan,
        ) {
            RequirementsCompatibility::Satisfy => {
                // println!("Window child is satisfying");
                let req = (!child_requirement.is_empty()).then(|| {
                    determine_satisfied_requirement_for_window(
                        parent_requirement.clone(),
                        child_requirement.clone(),
                        Arc::clone(&plan),
                        &mut soft_plans,
                    )
                });
                let mut result =
                    PushdownRequirementToChildrenResult::new(Some(vec![req]), soft_plans);

                result
            }
            RequirementsCompatibility::Compatible(adjusted) => {
                // println!("Window compatible");
                if let (RequiredInputOrdering::Soft(_), RequiredInputOrdering::Soft(_)) =
                    (parent_requirement, child_requirement.clone())
                {
                    let contains = soft_plans.iter().any(|p| Arc::ptr_eq(p, &plan));
                    if !contains {
                        soft_plans.push(Arc::clone(&plan));
                    }
                }

                // Check if Requirement has columns that are generated by window expressions
                let input_schema_len = child_plan.schema().fields.len();
                for sort_req in parent_requirement.to_vec() {
                    let columns = collect_columns(&sort_req.expr);
                    if columns.iter().any(|col| col.index() >= input_schema_len) {
                        // If parent requirements are more specific than output ordering
                        // of the window plan, then we can deduce that the parent expects
                        // an ordering from the columns created by window functions. If
                        // that's the case, we block the pushdown of sort operation.
                        if !plan.equivalence_properties().ordering_satisfy_requirement(
                            parent_requirement.lex_requirement(),
                        ) {
                            // println!("Extra field as sort op parent req: {:?}", parent_requirement);
                            // println!("Extra field as sort op output ordering: {:?}", plan.output_ordering());
                            return Ok(PushdownRequirementToChildrenResult::new(
                                None, soft_plans,
                            ));
                        }
                    }
                }

                PushdownRequirementToChildrenResult::new(Some(vec![adjusted]), soft_plans)
            }
            RequirementsCompatibility::NonCompatible => {
                // println!("Window non-compatible");
                PushdownRequirementToChildrenResult::new(None, soft_plans)
            }
        };
        match (parent_requirement, child_requirement) {
            (
                RequiredInputOrdering::Mixed((_, mixed_par)),
                RequiredInputOrdering::Mixed((_, mixed_child)),
            ) => {
                if !plan
                    .equivalence_properties()
                    .requirements_compatible(&mixed_par, &mixed_child)
                {
                    result = result.with_add_sort_requirement(
                        RequiredInputOrdering::Hard(mixed_par.clone()),
                    );
                }
            }
            _ => {}
        }
        Ok(result)
    } else if let Some(sort_exec) = plan.as_any().downcast_ref::<SortExec>() {
        let sort_req = RequiredInputOrdering::from(
            sort_exec
                .properties()
                .output_ordering()
                .cloned()
                .unwrap_or(LexOrdering::default()),
        );
        if sort_exec
            .properties()
            .eq_properties
            .requirements_compatible(
                parent_requirement.lex_requirement(),
                sort_req.lex_requirement(),
            )
        {
            debug_assert!(!parent_requirement.is_empty());
            Ok(PushdownRequirementToChildrenResult::new(
                Some(vec![Some(parent_requirement.clone())]),
                soft_plans,
            ))
        } else {
            Ok(PushdownRequirementToChildrenResult::new(None, soft_plans))
        }
    } else if plan.fetch().is_some()
        && plan.supports_limit_pushdown()
        && plan
            .maintains_input_order()
            .iter()
            .all(|maintain| *maintain)
    {
        let output_req = LexRequirement::from(
            plan.properties()
                .output_ordering()
                .cloned()
                .unwrap_or(LexOrdering::default()),
        );
        // Push down through operator with fetch when:
        // - requirement is aligned with output ordering
        // - it preserves ordering during execution
        if plan
            .properties()
            .eq_properties
            .requirements_compatible(parent_requirement.lex_requirement(), &output_req)
        {
            let req = (!parent_requirement.is_empty()).then(|| {
                RequiredInputOrdering::Hard(LexRequirement::new(
                    parent_requirement.to_vec(),
                ))
            });
            Ok(PushdownRequirementToChildrenResult::new(
                Some(vec![req]),
                soft_plans,
            ))
        } else {
            Ok(PushdownRequirementToChildrenResult::new(None, soft_plans))
        }
    } else if is_union(&plan) {
        // UnionExec does not have real sort requirements for its input. Here we change the adjusted_request_ordering to UnionExec's output ordering and
        // propagate the sort requirements down to correct the unnecessary descendant SortExec under the UnionExec
        let req = (!parent_requirement.is_empty()).then(|| parent_requirement.clone());
        Ok(PushdownRequirementToChildrenResult::new(
            Some(vec![req; plan.children().len()]),
            soft_plans,
        ))
    } else if let Some(smj) = plan.as_any().downcast_ref::<SortMergeJoinExec>() {
        // If the current plan is SortMergeJoinExec
        let left_columns_len = smj.left().schema().fields().len();
        let parent_required_expr =
            LexOrdering::from(parent_requirement.lex_requirement().clone());
        match expr_source_side(
            parent_required_expr.as_ref(),
            smj.join_type(),
            left_columns_len,
        ) {
            Some(JoinSide::Left) => Ok(PushdownRequirementToChildrenResult::new(
                try_pushdown_requirements_to_join(
                    smj,
                    parent_requirement.lex_requirement(),
                    parent_required_expr.as_ref(),
                    JoinSide::Left,
                )?,
                soft_plans,
            )),
            Some(JoinSide::Right) => {
                let right_offset =
                    smj.schema().fields.len() - smj.right().schema().fields.len();
                let new_right_required = shift_right_required(
                    parent_requirement.lex_requirement(),
                    right_offset,
                )?;
                let new_right_required_expr = LexOrdering::from(new_right_required);
                let res = try_pushdown_requirements_to_join(
                    smj,
                    parent_requirement.lex_requirement(),
                    new_right_required_expr.as_ref(),
                    JoinSide::Right,
                )?;
                Ok(PushdownRequirementToChildrenResult::new(res, soft_plans))
            }
            _ => {
                // Can not decide the expr side for SortMergeJoinExec, can not push down
                Ok(PushdownRequirementToChildrenResult::new(None, soft_plans))
            }
        }
    } else if maintains_input_order.is_empty()
        || !maintains_input_order.iter().any(|o| *o)
        || plan.as_any().is::<RepartitionExec>()
        || plan.as_any().is::<FilterExec>()
        // TODO: Add support for Projection push down
        || plan.as_any().is::<ProjectionExec>()
        || pushdown_would_violate_requirements(parent_requirement.lex_requirement(), plan.as_ref())
    {
        // If the current plan is a leaf node or can not maintain any of the input ordering, can not pushed down sort_push_down.
        // For RepartitionExec, we always choose to not push down the sort requirements even the RepartitionExec(input_partition=1) could maintain input ordering.
        // Pushing down is not beneficial
        Ok(PushdownRequirementToChildrenResult::new(None, soft_plans))
    } else if is_sort_preserving_merge(&plan) {
        let new_ordering =
            LexOrdering::from(parent_requirement.lex_requirement().clone());
        let mut spm_eqs = plan.equivalence_properties().clone();
        // Sort preserving merge will have new ordering, one requirement above is pushed down to its below.
        spm_eqs = spm_eqs.with_reorder(new_ordering);
        // Do not push-down through SortPreservingMergeExec when
        // ordering requirement invalidates requirement of sort preserving merge exec.
        if !spm_eqs.ordering_satisfy(&plan.output_ordering().cloned().unwrap_or_default())
        {
            Ok(PushdownRequirementToChildrenResult::new(None, soft_plans))
        } else {
            // Can push-down through SortPreservingMergeExec, because parent requirement is finer
            // than SortPreservingMergeExec output ordering.
            let req = (!parent_requirement.is_empty()).then(|| {
                RequiredInputOrdering::Hard(LexRequirement::new(
                    parent_requirement.to_vec(),
                ))
            });
            Ok(PushdownRequirementToChildrenResult::new(
                Some(vec![req]),
                soft_plans,
            ))
        }
    } else if let Some(hash_join) = plan.as_any().downcast_ref::<HashJoinExec>() {
        let res = handle_hash_join(hash_join, parent_requirement)?;
        Ok(PushdownRequirementToChildrenResult::new(res, soft_plans))
    } else {
        let res =
            handle_custom_pushdown(&plan, parent_requirement, maintains_input_order)?;
        Ok(PushdownRequirementToChildrenResult::new(res, soft_plans))
    }
    // TODO: Add support for Projection push down
}

/// Return true if pushing the sort requirements through a node would violate
/// the input sorting requirements for the plan
fn pushdown_would_violate_requirements(
    parent_required: &LexRequirement,
    child: &dyn ExecutionPlan,
) -> bool {
    child
        .required_input_ordering()
        .iter()
        .any(|child_required| {
            let Some(child_required) = child_required.as_ref() else {
                // no requirements, so pushing down would not violate anything
                return false;
            };
            // check if the plan's requirements would still e satisfied if we pushed
            // down the parent requirements
            child_required
                .lex_requirement()
                .iter()
                .zip(parent_required.iter())
                .all(|(c, p)| !c.compatible(p))
        })
}

/// Determine children requirements:
/// - If children requirements are more specific, do not push down parent
///   sort_push_down.
/// - If parent requirements are more specific, push down parent sort_push_down.
/// - If they are not compatible, need to add a sort.
fn determine_children_requirement(
    parent_required: &RequiredInputOrdering,
    child_requirement: &RequiredInputOrdering,
    child_plan: &Arc<dyn ExecutionPlan>,
) -> RequirementsCompatibility {
    if child_plan.equivalence_properties().requirements_compatible(
        child_requirement.lex_requirement(),
        parent_required.lex_requirement(),
    ) {
        // Child requirements are more specific, no need to push down.
        RequirementsCompatibility::Satisfy
    } else if child_plan.equivalence_properties().requirements_compatible(
        parent_required.lex_requirement(),
        child_requirement.lex_requirement(),
    ) {
        // Parent requirements are more specific, adjust child's requirements
        // and push down the new requirements:
        let adjusted = (!parent_required.lex_requirement().is_empty()).then(|| {
            let lex_req = LexRequirement::new(parent_required.to_vec());
            match (parent_required, child_requirement) {
                (RequiredInputOrdering::Soft(_), RequiredInputOrdering::Soft(_)) => {
                    RequiredInputOrdering::Soft(lex_req)
                }
                _ => RequiredInputOrdering::Hard(lex_req),
            }
        });
        RequirementsCompatibility::Compatible(adjusted)
    } else {
        RequirementsCompatibility::NonCompatible
    }
}

fn try_pushdown_requirements_to_join(
    smj: &SortMergeJoinExec,
    parent_required: &LexRequirement,
    sort_expr: &LexOrdering,
    push_side: JoinSide,
) -> Result<Option<Vec<Option<RequiredInputOrdering>>>> {
    let left_eq_properties = smj.left().equivalence_properties();
    let right_eq_properties = smj.right().equivalence_properties();
    let mut smj_required_orderings = smj.required_input_ordering();
    let right_requirement = smj_required_orderings.swap_remove(1);
    let left_requirement = smj_required_orderings.swap_remove(0);
    let left_ordering = &smj.left().output_ordering().cloned().unwrap_or_default();
    let right_ordering = &smj.right().output_ordering().cloned().unwrap_or_default();

    let (new_left_ordering, new_right_ordering) = match push_side {
        JoinSide::Left => {
            let left_eq_properties =
                left_eq_properties.clone().with_reorder(sort_expr.clone());
            if left_eq_properties.ordering_satisfy_requirement(
                left_requirement.unwrap_or_default().lex_requirement(),
            ) {
                // After re-ordering requirement is still satisfied
                (sort_expr, right_ordering)
            } else {
                return Ok(None);
            }
        }
        JoinSide::Right => {
            let right_eq_properties =
                right_eq_properties.clone().with_reorder(sort_expr.clone());
            if right_eq_properties.ordering_satisfy_requirement(
                right_requirement.unwrap_or_default().lex_requirement(),
            ) {
                // After re-ordering requirement is still satisfied
                (left_ordering, sort_expr)
            } else {
                return Ok(None);
            }
        }
        JoinSide::None => return Ok(None),
    };
    let join_type = smj.join_type();
    let probe_side = SortMergeJoinExec::probe_side(&join_type);
    let new_output_ordering = calculate_join_output_ordering(
        new_left_ordering,
        new_right_ordering,
        join_type,
        smj.on(),
        smj.left().schema().fields.len(),
        &smj.maintains_input_order(),
        Some(probe_side),
    );
    let mut smj_eqs = smj.properties().equivalence_properties().clone();
    // smj will have this ordering when its input changes.
    smj_eqs = smj_eqs.with_reorder(new_output_ordering.unwrap_or_default());
    let should_pushdown = smj_eqs.ordering_satisfy_requirement(parent_required);
    Ok(should_pushdown.then(|| {
        let mut required_input_ordering = smj.required_input_ordering();
        let new_req = Some(RequiredInputOrdering::from(sort_expr.clone()));
        match push_side {
            JoinSide::Left => {
                required_input_ordering[0] = new_req;
            }
            JoinSide::Right => {
                required_input_ordering[1] = new_req;
            }
            JoinSide::None => unreachable!(),
        }
        required_input_ordering
    }))
}

fn expr_source_side(
    required_exprs: &LexOrdering,
    join_type: JoinType,
    left_columns_len: usize,
) -> Option<JoinSide> {
    match join_type {
        JoinType::Inner
        | JoinType::Left
        | JoinType::Right
        | JoinType::Full
        | JoinType::LeftMark => {
            let all_column_sides = required_exprs
                .iter()
                .filter_map(|r| {
                    r.expr.as_any().downcast_ref::<Column>().map(|col| {
                        if col.index() < left_columns_len {
                            JoinSide::Left
                        } else {
                            JoinSide::Right
                        }
                    })
                })
                .collect::<Vec<_>>();

            // If the exprs are all coming from one side, the requirements can be pushed down
            if all_column_sides.len() != required_exprs.len() {
                None
            } else if all_column_sides
                .iter()
                .all(|side| matches!(side, JoinSide::Left))
            {
                Some(JoinSide::Left)
            } else if all_column_sides
                .iter()
                .all(|side| matches!(side, JoinSide::Right))
            {
                Some(JoinSide::Right)
            } else {
                None
            }
        }
        JoinType::LeftSemi | JoinType::LeftAnti => required_exprs
            .iter()
            .all(|e| e.expr.as_any().downcast_ref::<Column>().is_some())
            .then_some(JoinSide::Left),
        JoinType::RightSemi | JoinType::RightAnti => required_exprs
            .iter()
            .all(|e| e.expr.as_any().downcast_ref::<Column>().is_some())
            .then_some(JoinSide::Right),
    }
}

fn shift_right_required(
    parent_required: &LexRequirement,
    left_columns_len: usize,
) -> Result<LexRequirement> {
    let new_right_required = parent_required
        .iter()
        .filter_map(|r| {
            let col = r.expr.as_any().downcast_ref::<Column>()?;
            col.index().checked_sub(left_columns_len).map(|offset| {
                r.clone()
                    .with_expr(Arc::new(Column::new(col.name(), offset)))
            })
        })
        .collect::<Vec<_>>();
    if new_right_required.len() == parent_required.len() {
        Ok(LexRequirement::new(new_right_required))
    } else {
        plan_err!(
            "Expect to shift all the parent required column indexes for SortMergeJoin"
        )
    }
}

/// Handles the custom pushdown of parent-required sorting requirements down to
/// the child execution plans, considering whether the input order is maintained.
///
/// # Arguments
///
/// * `plan` - A reference to an `ExecutionPlan` for which the pushdown will be applied.
/// * `parent_required` - The sorting requirements expected by the parent node.
/// * `maintains_input_order` - A vector of booleans indicating whether each child
///   maintains the input order.
///
/// # Returns
///
/// Returns `Ok(Some(Vec<Option<LexRequirement>>))` if the sorting requirements can be
/// pushed down, `Ok(None)` if not. On error, returns a `Result::Err`.
fn handle_custom_pushdown(
    plan: &Arc<dyn ExecutionPlan>,
    parent_required: &RequiredInputOrdering,
    maintains_input_order: Vec<bool>,
) -> Result<Option<Vec<Option<RequiredInputOrdering>>>> {
    // If there's no requirement from the parent or the plan has no children, return early
    if parent_required.is_empty() || plan.children().is_empty() {
        return Ok(None);
    }

    // Collect all unique column indices used in the parent-required sorting expression
    let all_indices: HashSet<usize> = parent_required
        .lex_requirement()
        .iter()
        .flat_map(|order| {
            collect_columns(&order.expr)
                .iter()
                .map(|col| col.index())
                .collect::<HashSet<_>>()
        })
        .collect();

    // Get the number of fields in each child's schema
    let len_of_child_schemas: Vec<usize> = plan
        .children()
        .iter()
        .map(|c| c.schema().fields().len())
        .collect();

    // Find the index of the child that maintains input order
    let Some(maintained_child_idx) = maintains_input_order
        .iter()
        .enumerate()
        .find(|(_, m)| **m)
        .map(|pair| pair.0)
    else {
        return Ok(None);
    };

    // Check if all required columns come from the child that maintains input order
    let start_idx = len_of_child_schemas[..maintained_child_idx]
        .iter()
        .sum::<usize>();
    let end_idx = start_idx + len_of_child_schemas[maintained_child_idx];
    let all_from_maintained_child =
        all_indices.iter().all(|i| i >= &start_idx && i < &end_idx);

    // If all columns are from the maintained child, update the parent requirements
    if all_from_maintained_child {
        let sub_offset = len_of_child_schemas
            .iter()
            .take(maintained_child_idx)
            .sum::<usize>();
        // Transform the parent-required expression for the child schema by adjusting columns
        let updated_parent_req = parent_required
            .lex_requirement()
            .iter()
            .map(|req| {
                let child_schema = plan.children()[maintained_child_idx].schema();
                let updated_columns = Arc::clone(&req.expr)
                    .transform_up(|expr| {
                        if let Some(col) = expr.as_any().downcast_ref::<Column>() {
                            let new_index = col.index() - sub_offset;
                            Ok(Transformed::yes(Arc::new(Column::new(
                                child_schema.field(new_index).name(),
                                new_index,
                            ))))
                        } else {
                            Ok(Transformed::no(expr))
                        }
                    })?
                    .data;
                Ok(PhysicalSortRequirement::new(updated_columns, req.options))
            })
            .collect::<Result<Vec<_>>>()?;

        // Prepare the result, populating with the updated requirements for children that maintain order
        let result = maintains_input_order
            .iter()
            .map(|&maintains_order| {
                if maintains_order {
                    Some(
                        parent_required
                            .with_updated_requirements(updated_parent_req.clone()),
                    )
                } else {
                    None
                }
            })
            .collect();

        Ok(Some(result))
    } else {
        Ok(None)
    }
}

// For hash join we only maintain the input order for the right child
// for join type: Inner, Right, RightSemi, RightAnti
fn handle_hash_join(
    plan: &HashJoinExec,
    parent_required: &RequiredInputOrdering,
) -> Result<Option<Vec<Option<RequiredInputOrdering>>>> {
    // If there's no requirement from the parent or the plan has no children
    // or the join type is not Inner, Right, RightSemi, RightAnti, return early
    if parent_required.is_empty() || !plan.maintains_input_order()[1] {
        return Ok(None);
    }

    // Collect all unique column indices used in the parent-required sorting expression
    let all_indices: HashSet<usize> = parent_required
        .lex_requirement()
        .iter()
        .flat_map(|order| {
            collect_columns(&order.expr)
                .into_iter()
                .map(|col| col.index())
                .collect::<HashSet<_>>()
        })
        .collect();

    let column_indices = build_join_column_index(plan);
    let projected_indices: Vec<_> = if let Some(projection) = &plan.projection {
        projection.iter().map(|&i| &column_indices[i]).collect()
    } else {
        column_indices.iter().collect()
    };
    let len_of_left_fields = projected_indices
        .iter()
        .filter(|ci| ci.side == JoinSide::Left)
        .count();

    let all_from_right_child = all_indices.iter().all(|i| *i >= len_of_left_fields);

    // If all columns are from the right child, update the parent requirements
    if all_from_right_child {
        // Transform the parent-required expression for the child schema by adjusting columns
        let updated_parent_req = parent_required
            .lex_requirement()
            .iter()
            .map(|req| {
                let child_schema = plan.children()[1].schema();
                let updated_columns = Arc::clone(&req.expr)
                    .transform_up(|expr| {
                        if let Some(col) = expr.as_any().downcast_ref::<Column>() {
                            let index = projected_indices[col.index()].index;
                            Ok(Transformed::yes(Arc::new(Column::new(
                                child_schema.field(index).name(),
                                index,
                            ))))
                        } else {
                            Ok(Transformed::no(expr))
                        }
                    })?
                    .data;
                Ok(PhysicalSortRequirement::new(updated_columns, req.options))
            })
            .collect::<Result<Vec<_>>>()?;

        // Populating with the updated requirements for children that maintain order
        Ok(Some(vec![
            None,
            Some(parent_required.with_updated_requirements(updated_parent_req)),
        ]))
    } else {
        Ok(None)
    }
}

// this function is used to build the column index for the hash join
// push down sort requirements to the right child
fn build_join_column_index(plan: &HashJoinExec) -> Vec<ColumnIndex> {
    let map_fields = |schema: SchemaRef, side: JoinSide| {
        schema
            .fields()
            .iter()
            .enumerate()
            .map(|(index, _)| ColumnIndex { index, side })
            .collect::<Vec<_>>()
    };

    match plan.join_type() {
        JoinType::Inner | JoinType::Right => {
            map_fields(plan.left().schema(), JoinSide::Left)
                .into_iter()
                .chain(map_fields(plan.right().schema(), JoinSide::Right))
                .collect::<Vec<_>>()
        }
        JoinType::RightSemi | JoinType::RightAnti => {
            map_fields(plan.right().schema(), JoinSide::Right)
        }
        _ => unreachable!("unexpected join type: {}", plan.join_type()),
    }
}

/// Define the Requirements Compatibility
#[derive(Debug)]
enum RequirementsCompatibility {
    /// Requirements satisfy
    Satisfy,
    /// Requirements compatible
    Compatible(Option<RequiredInputOrdering>),
    /// Requirements not compatible
    NonCompatible,
}
