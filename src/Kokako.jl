#  Copyright 2018, Oscar Dowson.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

module Kokako

import Reexport
Reexport.@reexport using JuMP

const MOI = JuMP.MOI

import Printf, JSON, Random, RecipesBase, Statistics, TimerOutputs

export @stageobjective

# Modelling interface.
include("user_interface.jl")

# SDDP related modular utilities.
include("plugins/risk_measures.jl")
include("plugins/sampling_schemes.jl")
include("plugins/bellman_functions.jl")
include("plugins/stopping_rules.jl")

# Printing utilities.
include("print.jl")

# The core SDDP code.
include("sddp.jl")

# Visualization utilities.
include("visualization/publication_plot.jl")
include("visualization/spaghetti_plot.jl")

end
