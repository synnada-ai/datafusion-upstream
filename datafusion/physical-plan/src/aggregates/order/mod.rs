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

use datafusion_common::plan_datafusion_err;
use datafusion_common::Result;
use datafusion_expr::EmitTo;

mod full;

use crate::InputOrderMode;
pub(crate) use full::GroupOrderingFull;

/// Ordering information for each group in the hash table
#[derive(Debug)]
pub(crate) enum GroupOrdering {
    /// Groups are not ordered
    None,
    /// Groups are entirely contiguous,
    Full(GroupOrderingFull),
}

impl GroupOrdering {
    /// Create a `GroupOrdering` for the the specified ordering
    pub fn try_new(mode: &InputOrderMode) -> Result<Self> {
        match mode {
            InputOrderMode::Linear => Ok(GroupOrdering::None),
            InputOrderMode::PartiallySorted(_order_indices) => Err(plan_datafusion_err!(
                "AggregateExec cannot receive input in the PartiallySorted mode."
            )),
            InputOrderMode::Sorted => Ok(GroupOrdering::Full(GroupOrderingFull::new())),
        }
    }

    // How many groups be emitted, or None if no data can be emitted
    pub fn emit_to(&self) -> Option<EmitTo> {
        match self {
            GroupOrdering::None => None,
            GroupOrdering::Full(full) => full.emit_to(),
        }
    }

    /// Updates the state the input is done
    pub fn input_done(&mut self) {
        match self {
            GroupOrdering::None => {}
            GroupOrdering::Full(full) => full.input_done(),
        }
    }

    /// remove the first n groups from the internal state, shifting
    /// all existing indexes down by `n`
    pub fn remove_groups(&mut self, n: usize) {
        match self {
            GroupOrdering::None => {}
            GroupOrdering::Full(full) => full.remove_groups(n),
        }
    }

    /// Called when new groups are added in a batch
    ///
    /// * `total_num_groups`: total number of groups (so max
    /// group_index is total_num_groups - 1).
    ///
    /// * `group_values`: group key values for *each row* in the batch
    ///
    /// * `group_indices`: indices for each row in the batch
    ///
    /// * `hashes`: hash values for each row in the batch
    pub fn new_groups(&mut self, total_num_groups: usize) -> Result<()> {
        match self {
            GroupOrdering::None => {}
            GroupOrdering::Full(full) => {
                full.new_groups(total_num_groups);
            }
        };
        Ok(())
    }

    /// Return the size of memory used by the ordering state, in bytes
    pub(crate) fn size(&self) -> usize {
        std::mem::size_of::<Self>()
            + match self {
                GroupOrdering::None => 0,
                GroupOrdering::Full(full) => full.size(),
            }
    }
}
