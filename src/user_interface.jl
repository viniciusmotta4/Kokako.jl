#  Copyright 2018, Oscar Dowson.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

struct Graph{T}
    # The root node of the policy graph.
    root_node::T
    # nodes[x] returns a vector of the children of node x and their
    # probabilities.
    nodes::Dict{T, Vector{Tuple{T, Float64}}}
    # A partition of the nodes into ambiguity sets.
    belief_partition::Vector{Vector{T}}

    function Graph(root_node::T) where T
        return new{T}(
            root_node,
            Dict{T, Vector{Tuple{T, Float64}}}(
                root_node => Tuple{T, Float64}[]
            ),
            Vector{T}[]
        )
    end
end

function validate_graph(graph)
    for (node, children) in graph.nodes
        if length(children) > 0
            probability = sum(child[2] for child in children)
            if !(0.0 <= probability <= 1.0)
                error("Probability on edges leaving node $(node) sum to " *
                      "$(probability), but this must be in [0.0, 1.0]")
            end
        end
    end
    if length(graph.belief_partition) > 0
        # The -1 accounts for the root node, which shouldn't be in the
        # partition.
        if graph.root_node in union(graph.belief_partition...)
            error("Belief partition $(graph.belief_partition) cannot contain " *
                  "the root node $(graph.root_node).")
        end
        if length(graph.nodes) - 1 != length(union(graph.belief_partition...))
            error("Belief partition $(graph.belief_partition) does not form a" *
                  " valid partition of the nodes in the graph.")
        end
    end
end

"""
    add_node(graph::Graph{T}, node::T) where T

Add a node to the graph `graph`.

### Examples

    add_node(graph, :A)
"""
function add_node(graph::Graph{T}, node::T) where T
    if haskey(graph.nodes, node) || node == graph.root_node
        error("Node $(node) already exists!")
    end
    graph.nodes[node] = Tuple{T, Float64}[]
    return
end
function add_node(graph::Graph{T}, node) where T
    error("Unable to add node $(node). Nodes must be of type $(T).")
end

"""
    add_edge(graph::Graph{T}, edge::Pair{T, T}, probability::Float64) where T

Add an edge to the graph `graph`.

### Examples

    add_edge(graph, 1 => 2, 0.9)
    add_edge(graph, :root => :A, 1.0)
"""
function add_edge(graph::Graph{T}, edge::Pair{T, T},
                  probability::Float64) where T
    (parent, child) = edge
    if !(parent == graph.root_node || haskey(graph.nodes, parent))
        error("Node $(parent) does not exist.")
    elseif !haskey(graph.nodes, child)
        error("Node $(child) does not exist.")
    elseif child == graph.root_node
        error("Cannot have an edge entering the root node.")
    else
        push!(graph.nodes[parent], (child, probability))
    end
    return
end

"""
    add_partition(graph::Graph{T}, set::Vector{T})

Add `set` to the belief partition of `graph`.

### Examples

    graph = LinearGraph(3)
    add_partition(graph, [1, 2])
    add_partition(graph, [3])
"""
function add_partition(graph::Graph{T}, set::Vector{T}) where T
    push!(graph.belief_partition, set)
    return
end

function Graph(root_node::T, nodes::Vector{T},
               edges::Vector{Tuple{Pair{T, T}, Float64}};
               belief_partition::Vector{Vector{T}} = Vector{T}[]) where T
    graph = Graph(root_node)
    add_node.(Ref(graph), nodes)
    for (edge, probability) in edges
        add_edge(graph, edge, probability)
    end
    add_partition.(Ref(graph), belief_partition)
    return graph
end

"""
    LinearGraph(stages::Int)
"""
function LinearGraph(stages::Int)
    edges = Tuple{Pair{Int, Int}, Float64}[]
    for t in 1:stages
        push!(edges, (t - 1 => t, 1.0))
    end
    return Graph(0, collect(1:stages), edges)
end

"""
    MarkovianGraph(transition_matrices::Vector{Matrix{Float64}})
"""
function MarkovianGraph(transition_matrices::Vector{Matrix{Float64}})
    if size(transition_matrices[1], 1) != 1
        error("Expected the first transition matrix to be of size (1, N). It " *
              "is of size $(size(transition_matrices[1])).")
    end
    node_type = Tuple{Int, Int}
    root_node = (0, 1)
    nodes = node_type[]
    edges = Tuple{Pair{node_type, node_type}, Float64}[]
    for (stage, transition) in enumerate(transition_matrices)
        if !all(transition .>= 0.0)
            error("Entries in the transition matrix must be non-negative.")
        end
        if !all(0.0 .<= sum(transition; dims=2) .<= 1.0)
            error("Rows in the transition matrix must sum to between 0.0 and " *
                  "1.0.")
        end
        if stage > 1
            if size(transition_matrices[stage-1], 2) != size(transition, 1)
                error("Transition matrix for stage $(stage) is the wrong size.")
            end
        end
        for markov_state in 1:size(transition, 2)
            push!(nodes, (stage, markov_state))
        end
        for markov_state in 1:size(transition, 2)
            for last_markov_state in 1:size(transition, 1)
                probability = transition[last_markov_state, markov_state]
                if 0.0 < probability <= 1.0
                    push!(edges, (
                        (stage - 1, last_markov_state) => (stage, markov_state),
                        probability
                    ))
                end
            end
        end
    end
    return Graph(root_node, nodes, edges)
end

"""
    MarkovianGraph(; stages::Int,
                   transition_matrix::Matrix{Float64},
                   root_node_transition::Vector{Float64})

Construct a Markovian graph object.
"""
function MarkovianGraph(; stages::Int = 1,
                        transition_matrix::Matrix{Float64}=[1.0],
                        root_node_transition::Vector{Float64}=[1.0])
    return MarkovianGraph(
        vcat([reshape(root_node_transition, 1, length(root_node_transition))],
             [transition_matrix for stage in 1:(stages - 1)])
    )
end

struct Noise{T}
    # The noise term.
    term::T
    # The probability of sampling the noise term.
    probability::Float64
end

struct State{T}
    # The incoming state variable.
    in::T
    # The outgoing state variable.
    out::T
end

mutable struct Node{T}
    # The index of the node in the policy graph.
    index::T
    # The JuMP subproblem.
    subproblem::JuMP.Model
    # A vector of the child nodes.
    children::Vector{Noise{T}}
    # A vector of the discrete stagewise-independent noise terms.
    noise_terms::Vector{Noise}
    # A function parameterize(model::JuMP.Model, noise) that modifies the JuMP
    # model based on the observation of the noise.
    parameterize::Function  # TODO(odow): make this a concrete type?
    # A list of the state variables in the model.
    states::Dict{Symbol, State{JuMP.VariableRef}}
    # Stage objective
    stage_objective  # TODO(odow): make this a concrete type?
    stage_objective_set::Bool
    # Bellman function
    bellman_function  # TODO(odow): make this a concrete type?
end

struct PolicyGraph{T}
    # Must be MOI.MinSense or MOI.MaxSense
    objective_sense::MOI.OptimizationSense
    # Children of the root node. child => probability.
    root_children::Vector{Noise{T}}
    # Starting value of the state variables.
    initial_root_state::Dict{Symbol, Float64}
    # All nodes in the graph.
    nodes::Dict{T, Node{T}}
    # Belief partition.
    belief_partition::Vector{Set{T}}

    function PolicyGraph(T, sense::Symbol)
        optimization_sense = if sense == :Min
            MOI.MinSense
        elseif sense == :Max
            MOI.MaxSense
        else
            error("The optimization sense must be :Min or :Max. It is $(sense).")
        end
        return new{T}(optimization_sense, Noise{T}[], Dict{Symbol, Float64}(),
            Dict{T, Node{T}}(), Set{T}[])
    end
end

# So we can query nodes in the graph as graph[node].
function Base.getindex(graph::PolicyGraph{T}, index::T) where T
    return graph.nodes[index]
end

function get_subproblem(graph::PolicyGraph{T}, index::T) where T
    return graph[index].subproblem::JuMP.Model
end

# Work around different JuMP modes (Automatic / Manual / Direct).
function construct_subproblem(optimizer_factory, direct_mode::Bool)
    subproblem = if direct_mode
        instance = optimizer_factory.constructor(
            optimizer_factory.args...; optimizer_factory.kwargs...)
        JuMP.direct_model(instance)
    else
        JuMP.Model(optimizer_factory)
    end
    return subproblem
end

# Work around different JuMP modes (Automatic / Manual / Direct).
function construct_subproblem(optimizer_factory::Nothing, direct_mode::Bool)
    if direct_mode
        error("You must specify an optimizer in the form:\n" *
              "    with_optimizer(Module.Opimizer, args...) if " *
              "direct_mode=true.")
    end
    return JuMP.Model()
end

# Storage for belief-related things.
# TODO(odow): incorporate this into `::Node`?
struct BeliefState{T}
    partition_index::Int
    belief::Dict{T, Float64}
    μ::Dict{T, JuMP.VariableRef}
    updater::Function
end

"""
    PolicyGraph(builder::Function, graph::Graph{T};
                bellman_function = AverageCut,
                optimizer = nothing,
                direct_mode = true) where T

Construct a a policy graph based on the graph structure of `graph`. (See `Graph`
for details.)

# Example

    function builder(subproblem::JuMP.Model, index)
        # ... subproblem definition ...
    end
    model = PolicyGraph(builder, graph;
                        bellman_function = AverageCut,
                        optimizer = with_optimizer(GLPK.Optimizer),
                        direct_mode = false)

Or, using the Julia `do ... end` syntax:

    model = PolicyGraph(graph;
                        bellman_function = AverageCut,
                        optimizer = with_optimizer(GLPK.Optimizer),
                        direct_mode = true) do subproblem, index
        # ... subproblem definitions ...
    end
"""
function PolicyGraph(builder::Function, graph::Graph{T};
                     sense = :Min,
                     bellman_function = AverageCut(),
                     optimizer = nothing,
                     direct_mode = true) where {T}
    # Spend a one-off cost validating the graph.
    validate_graph(graph)
    # Construct a basic policy graph. We will add to it in the remainder of this
    # function.
    policy_graph = PolicyGraph(T, sense)
    # Initialize nodes.
    for (node_index, children) in graph.nodes
        if node_index == graph.root_node
            continue
        end
        subproblem = construct_subproblem(optimizer, direct_mode)
        node = Node(
            node_index,
            subproblem,
            Noise{T}[],
            Noise[],
            (ω) -> nothing,
            Dict{Symbol, State{JuMP.VariableRef}}(),
            nothing,
            false,
            # Delay initializing the bellman function until later so that it can
            # use information about the children and number of
            # stagewise-independent noise realizations.
            nothing

        )
        subproblem.ext[:kokako_policy_graph] = policy_graph
        policy_graph.nodes[node_index] = subproblem.ext[:kokako_node] = node
        builder(subproblem, node_index)
        # Add a dummy noise here so that all nodes have at least one noise term.
        if length(node.noise_terms) == 0
            push!(node.noise_terms, Noise(nothing, 1.0))
        end
    end
    # Loop back through and add the arcs/children.
    for (node_index, children) in graph.nodes
        if node_index == graph.root_node
            continue
        end
        node = policy_graph.nodes[node_index]
        for (child, probability) in children
            push!(node.children, Noise(child, probability))
        end
        # Intialize the bellman function. (See note in creation of Node above.)
        node.bellman_function = initialize_bellman_function(
            bellman_function, policy_graph, node)
    end
    # Add root nodes
    for (child, probability) in graph.nodes[graph.root_node]
        push!(policy_graph.root_children, Noise(child, probability))
    end

    # Pre-compute the function `belief_updater`. See `construct_belief_update`
    # for details.
    belief_updater = construct_belief_update(
        policy_graph, Set.(graph.belief_partition))
    # Initialize a belief dictionary (containing one element for each node in
    # the graph).
    belief = Dict{T, Float64}(keys(graph.nodes) .=> 0.0)
    delete!(belief, graph.root_node)
    # Now for each element in the partition...
    for (partition_index, partition) in enumerate(graph.belief_partition)
        # Store the partition in the `policy_graph` object.
        push!(policy_graph.belief_partition, Set(partition))
        # Then for each node in the partition.
        for node_index in partition
            # Get the `::Node` object.
            node = policy_graph[node_index]
            # Add the dual variable μ for the cut:
            # θ ≥ α + <β, x> - <b, μ>
            # We need one variable for each non-zero belief state.
            # TODO: lipschitz bounds: `|μ|∞ ≤ L`? Ideally we want these to be
            # node-dependent.
            μ = @variable(node.subproblem, [n in partition],
                lower_bound = -1e6,
                upper_bound = 1e6,
                container = Dict
            )
            # Attach the belief state as an extension.
            # TODO(odow): make the belief state part of the `::Node` object.
            node.subproblem.ext[:kokako_belief] = BeliefState{T}(
                partition_index,
                copy(belief),
                μ,
                belief_updater
            )
        end
    end
    return policy_graph
end

# Internal function: helper to get the node given a subproblem.
function get_node(subproblem::JuMP.Model)
    return subproblem.ext[:kokako_node]::Node
end

# Internal functino: helper to get the policy graph given a subproblem.
function get_policy_graph(subproblem::JuMP.Model)
    return subproblem.ext[:kokako_policy_graph]::PolicyGraph
end

"""
    parameterize(modify::Function,
                 subproblem::JuMP.Model,
                 realizations::Vector{T},
                 probability::Vector{Float64} = fill(1.0 / length(realizations))
                     ) where T

Add a parameterization function `modify` to `subproblem`. The `modify` function
takes one argument and modifies `subproblem` based on the realization of the
noise sampled from `realizations` with corresponding probabilities
`probability`.

In order to conduct an out-of-sample simulation, `modify` should accept
arguments that are not in realizations (but still of type T).

# Example

    Kokako.parameterize(subproblem, [1, 2, 3], [0.4, 0.3, 0.3]) do ω
        JuMP.set_upper_bound(x, ω)
    end
"""
function parameterize(modify::Function,
                      subproblem::JuMP.Model,
                      realizations::AbstractVector{T},
                      probability::AbstractVector{Float64} =
                          fill(1.0 / length(realizations), length(realizations))
                          ) where T
    node = get_node(subproblem)
    if length(node.noise_terms) != 0
        error("Duplicate calls to Kokako.parameterize detected. Only " *
              "a subproblem at most one time.")
    end
    for (realization, prob) in zip(realizations, probability)
        push!(node.noise_terms, Noise(realization, prob))
    end
    node.parameterize = modify
    return
end

"""
    set_stage_objective(subproblem::JuMP.Model, stage_objective)

Set the stage-objective of `subproblem` to `stage_objective`.

# Example

    Kokako.set_stage_objective(subproblem, 2x + 1)
"""
function set_stage_objective(subproblem::JuMP.Model, stage_objective)
    node = get_node(subproblem)
    node.stage_objective = stage_objective
    node.stage_objective_set = false
    return
end

macro stageobjective(subproblem, expr)
    code = quote
        set_stage_objective(
            $(esc(subproblem)),
            $(Expr(:macrocall,
                Symbol("@expression"),
                :LineNumber,
                esc(subproblem),
                esc(expr)
            ))
        )
    end
    return code
end

# ============================================================================ #
#
#   Code to implement a JuMP variable extension.
#
#   Usage:
#   julia> @variable(subproblem, 0 <= x[i=1:2] <= i,
#              Kokako.State, initial_value = i)
#
#   julia> x
#   2-element Array{State{VariableRef},1}:
#     State(x[1]_in,x[1]_out)
#     State(x[2]_in,x[2]_out)
#
#   julia> x[1].in
#   x[1]_in
#
#   julia> typeof(x[1].in)
#   VariableRef
#
#   julia> x[2].out
#   x[2]_out
#
#   Assuming subproblem has been solved, and there exists a primal solution
#   julia> x_values = JuMP.value.(x)
#   2-element Array{State{Float64},1}:
#     State(0.0,1.0)
#     State(1.2,3.0)
#
#   julia> x_values[1].out
#   1.0
# ============================================================================ #

struct StateInfo
    in::JuMP.VariableInfo
    out::JuMP.VariableInfo
    initial_value::Float64
end

function JuMP.build_variable(
        _error::Function, info::JuMP.VariableInfo, ::Type{State};
        initial_value = NaN,
        kwargs...)
    if isnan(initial_value)
        _error("When creating a state variable, you must set the " *
               "`initial_value` keyword to the value of the state variable at" *
               " the root node.")
    end
    return StateInfo(
        JuMP.VariableInfo(
            false, NaN,  # lower bound
            false, NaN,  # upper bound
            false, NaN,  # fixed value
            false, NaN,  # start value
            false, false # binary and integer
        ),
        info,
        initial_value
    )
end

function JuMP.add_variable(
        subproblem::JuMP.Model, state_info::StateInfo, name::String)
    state = State(
        JuMP.add_variable(
            subproblem, JuMP.ScalarVariable(state_info.in), name * "_in"),
        JuMP.add_variable(
            subproblem, JuMP.ScalarVariable(state_info.out), name * "_out")
    )
    node = get_node(subproblem)
    sym_name = Symbol(name)
    if haskey(node.states, sym_name)
        error("The state $(sym_name) already exists.")
    end
    node.states[sym_name] = state
    graph = get_policy_graph(subproblem)
    graph.initial_root_state[sym_name] = state_info.initial_value
    return state
end

JuMP.variable_type(model::JuMP.Model, ::Type{State}) = State

function JuMP.value(state::State{JuMP.VariableRef})
    return State(JuMP.value(state.in), JuMP.value(state.out))
end

"""
    construct_belief_update(graph::PolicyGraph{T}, partition::Vector{Set{T}})

Returns a function that calculates the belief update.
    construct_belief_update(
        incoming_belief::Dict{T, Float64},
        observed_partition::Int,
        observed_noise
    )::Dict{T, Float64}

We use Bayes theorem: P(X′ | Y) = P(Y | X′) × P(X′) / P(Y), where P(Xᵢ′ | Y) is
the probability of being in node i given the observation of ω.
P(Xⱼ′) = ∑ᵢ P(Xᵢ) × Φᵢⱼ
P(Y|Xᵢ′) = P(ω ∈ Ωᵢ)
P(Y) = ∑ᵢ P(Xᵢ′) × P(ω ∈ Ωᵢ)
"""
function construct_belief_update(
    graph::Kokako.PolicyGraph{T}, partition::Vector{Set{T}}) where T
    # TODO: check that partition is proper.
    # TODO: throw errors for invalid belief? Or do we just assume that we can
    #     never be given an invalid belief.
    Φ = Kokako.build_Φ(graph)  # Dict{Tuple{T, T}, Float64}
    Ω = Dict{T, Dict{Any, Float64}}()
    for (index, node) in graph.nodes
        Ω[index] = Dict{Any, Float64}()
        for noise in node.noise_terms
            Ω[index][noise.term] = noise.probability
        end
    end
    function belief_updater(
        outgoing_belief::Dict{T, Float64},
        incoming_belief::Dict{T, Float64},
        observed_partition::Int,
        observed_noise)::Dict{T, Float64}
        # P(Y) = ∑ᵢ Xᵢ × ∑ⱼ P(i->j) × P(ω ∈ Ωⱼ)
        PY = 0.0
        for (node_i, belief) in incoming_belief
            probability = 0.0
            for (node_j, Ωj) in Ω
                p_ij = get(Φ, (node_i, node_j), 0.0)
                p_ω = get(Ωj, observed_noise, 0.0)
                probability += p_ij * p_ω
            end
            PY += belief * probability
        end
        # Now update each belief.
        for (node_i, belief) in incoming_belief
            PX = sum(belief * get(Φ, (node_j, node_i), 0.0)
                for (node_j, belief) in incoming_belief)
            PY_X = 0.0
            if node_i in partition[observed_partition]
                PY_X += get(Ω[node_i], observed_noise, 0.0)
            end
            outgoing_belief[node_i] = PY_X * PX / PY
        end
        return outgoing_belief
    end
    return belief_updater
end
