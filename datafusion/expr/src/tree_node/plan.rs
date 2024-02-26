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

//! Tree node implementation for logical plan

use crate::LogicalPlan;

use datafusion_common::tree_node::{
    Transformed, TreeNode, TreeNodeRecursion, TreeNodeVisitor,
};
use datafusion_common::{handle_visit_recursion_down, handle_visit_recursion_up, Result};

impl TreeNode for LogicalPlan {
    fn apply<F: FnMut(&Self) -> Result<TreeNodeRecursion>>(
        &self,
        f: &mut F,
    ) -> Result<TreeNodeRecursion> {
        // Compared to the default implementation, we need to invoke
        // [`Self::apply_subqueries`] before visiting its children
        handle_visit_recursion_down!(f(self)?);
        self.apply_subqueries(f)?;
        self.apply_children(&mut |n| n.apply(f))
    }

    /// To use, define a struct that implements the trait [`TreeNodeVisitor`] and then invoke
    /// [`LogicalPlan::visit`].
    ///
    /// For example, for a logical plan like:
    ///
    /// ```text
    /// Projection: id
    ///    Filter: state Eq Utf8(\"CO\")\
    ///       CsvScan: employee.csv projection=Some([0, 3])";
    /// ```
    ///
    /// The sequence of visit operations would be:
    /// ```text
    /// visitor.pre_visit(Projection)
    /// visitor.pre_visit(Filter)
    /// visitor.pre_visit(CsvScan)
    /// visitor.post_visit(CsvScan)
    /// visitor.post_visit(Filter)
    /// visitor.post_visit(Projection)
    /// ```
    fn visit<V: TreeNodeVisitor<Node = Self>>(
        &self,
        visitor: &mut V,
    ) -> Result<TreeNodeRecursion> {
        // Compared to the default implementation, we need to invoke
        // [`Self::visit_subqueries`] before visiting its children
        handle_visit_recursion_down!(visitor.f_down(self)?);
        self.visit_subqueries(visitor)?;
        handle_visit_recursion_up!(self.apply_children(&mut |n| n.visit(visitor))?);
        visitor.f_up(self)
    }

    fn apply_children<F: FnMut(&Self) -> Result<TreeNodeRecursion>>(
        &self,
        f: &mut F,
    ) -> Result<TreeNodeRecursion> {
        let mut tnr = TreeNodeRecursion::Continue;
        for child in self.inputs() {
            tnr = f(child)?;
            handle_visit_recursion_down!(tnr)
        }
        Ok(tnr)
    }

    fn map_children<F>(self, f: F) -> Result<Transformed<Self>>
    where
        F: FnMut(Self) -> Result<Transformed<Self>>,
    {
        let old_children = self.inputs();
        let new_children = old_children
            .iter()
            .map(|&c| c.clone())
            .map(f)
            .collect::<Result<Vec<_>>>()?;

        // if any changes made, make a new child
        if old_children
            .into_iter()
            .zip(new_children.iter())
            .any(|(c1, c2)| c1 != &c2.data)
        {
            self.with_new_exprs(
                self.expressions(),
                new_children.into_iter().map(|child| child.data).collect(),
            )
            .map(Transformed::yes)
        } else {
            Ok(Transformed::no(self))
        }
    }
}
