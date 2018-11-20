#  Copyright 2018, Oscar Dowson.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

using Kokako, Test, GLPK

@testset "Basic Graphs" begin
    @testset "LinearGraph" begin
        graph = Kokako.LinearGraph(5)
        @test graph.root_node == 0
        for stage in 0:4
            @test haskey(graph.nodes, stage)
            @test graph.nodes[stage] == [(stage + 1, 1.0)]
        end
        @test haskey(graph.nodes, 5)
        @test graph.nodes[5] == Tuple{Int, Float64}[]
        @test length(graph.belief_partition) == 0
    end

    @testset "MarkovianGraph" begin
        @testset "Error cases" begin
            # Not root transition matrix.
            @test_throws Exception Kokako.MarkovianGraph([[0.5 0.5; 0.5 0.5]])
            # Negative probability.
            @test_throws Exception Kokako.MarkovianGraph([[-0.5 0.75]])
            # Proability sums to greater than 1.
            @test_throws Exception Kokako.MarkovianGraph([[0.8 0.8]])
            # Mis-matched dimensions.
            @test_throws Exception Kokako.MarkovianGraph([
                [0.1 0.2 0.7], [0.5 0.5; 0.5 0.5]
            ])
        end
        @testset "keyword vs list" begin
            graph_1 = Kokako.MarkovianGraph(
                stages = 2,
                transition_matrix = [0.4 0.6; 0.25 0.75],
                root_node_transition = [0.7, 0.3]
            )
            graph_2 = Kokako.MarkovianGraph([
                [0.7 0.3], [0.4 0.6; 0.25 0.75]
            ])
            @test graph_1.root_node == graph_2.root_node
            @test graph_1.nodes == graph_2.nodes
            @test length(graph_1.belief_partition) == 0
            @test length(graph_2.belief_partition) == 0
        end
    end

    @testset "Graph" begin
        @testset "Construct Graph" begin
            graph = Kokako.Graph(:root)
            @test graph.root_node == :root
            @test collect(keys(graph.nodes)) == [:root]
        end
        @testset "Add node" begin
            graph = Kokako.Graph(:root)
            Kokako.add_node(graph, :x)
            @test collect(keys(graph.nodes)) == [:root, :x]
        end
        @testset "Add duplicate node" begin
            graph = Kokako.Graph(:root)
            Kokako.add_node(graph, :x)
            @test_throws Exception Kokako.add_node(graph, :x)
        end
        @testset "Add edge" begin
            graph = Kokako.Graph(:root)
            Kokako.add_node(graph, :x)
            Kokako.add_edge(graph, :root => :x, 1.0)
            @test haskey(graph.nodes, :root)
            @test graph.nodes[:root] == [(:x, 1.0)]
        end
        @testset "Add edge of wrong type" begin
            graph = Kokako.Graph(:root)
            @test_throws Exception Kokako.add_node(graph, 1)
        end
        @testset "Add edge to missing node" begin
            graph = Kokako.Graph(:root)
            Kokako.add_node(graph, :x)
            @test_throws Exception Kokako.add_edge(graph, :x => :y, 1.0)
            @test_throws Exception Kokako.add_edge(graph, :y => :x, 1.0)
        end
        @testset "Add edge to root" begin
            graph = Kokako.Graph(:root)
            Kokako.add_node(graph, :x)
            @test_throws Exception Kokako.add_edge(graph, :x => :root, 1.0)
        end
        @testset "Invalid probability" begin
            graph = Kokako.Graph(:root)
            Kokako.add_node(graph, :x)
            Kokako.add_edge(graph, :root => :x, 0.5)
            Kokako.add_edge(graph, :root => :x, 0.75)
            @test_throws Exception Kokako.validate_graph(graph)
        end
        @testset "Belief partition" begin
            graph = Kokako.Graph(:root)
            Kokako.add_node(graph, :x)
            Kokako.add_node(graph, :y)
            Kokako.add_partition(graph, [:x])
            Kokako.add_partition(graph, [:y])
            @test graph.belief_partition == [ [:x], [:y] ]

            graph = Kokako.Graph(:root, [:x, :y], [
                (:root => :x, 0.5),
                (:root => :y, 0.5),
                ],
                belief_partition = [ [:x, :y] ]
            )
            @test graph.belief_partition == [ [:x, :y] ]

            graph = Kokako.Graph(:root, [:x, :y], [
                (:root => :x, 0.5),
                (:root => :y, 0.5),
                ]
            )
            @test length(graph.belief_partition) == 0
        end
    end
end

@testset "PolicyGraph constructor" begin
    @testset "LinearGraph" begin
        model = Kokako.PolicyGraph(Kokako.LinearGraph(2),
                                   direct_mode=false) do node, stage
        end

        @test_throws Exception Kokako.PolicyGraph(Kokako.LinearGraph(2)
                                                      ) do node, stage
        end

        model = Kokako.PolicyGraph(Kokako.LinearGraph(2),
                                   optimizer = with_optimizer(GLPK.Optimizer)
                                       ) do node, stage
        end
    end

    @testset "MarkovianGraph" begin
        graph = Kokako.MarkovianGraph([
                ones(Float64, (1, 1)),
                [0.5 0.5],
                [0.5 0.5; 0.3 0.4],
                [0.5 0.5; 0.3 0.4],
                [0.5 0.5; 0.3 0.4]
            ]
        )
        model = Kokako.PolicyGraph(graph, direct_mode = false) do node, stage
        end
    end

    @testset "General" begin
        graph = Kokako.Graph(
            :root,
            [:stage_1, :stage_2, :stage_3],
            [
                (:root => :stage_1, 1.0),
                (:stage_1 => :stage_2, 1.0),
                (:stage_2 => :stage_3, 1.0),
                (:stage_3 => :stage_1, 0.9)
            ]
        )
        model = Kokako.PolicyGraph(graph, direct_mode = false) do node, stage
        end
    end
end

@testset "Kokako.State" begin
    model = Kokako.PolicyGraph(Kokako.LinearGraph(2),
                               direct_mode = false) do node, stage
        @variable(node, x, Kokako.State, initial_value = 0)
    end
    for stage in 1:2
        node = model[stage]
        @test haskey(node.states, :x)
        @test length(keys(node.states)) == 1
        @test node.states[:x] == node.subproblem[:x]
    end
end

@testset "Kokako.parameterize" begin
    model = Kokako.PolicyGraph(Kokako.LinearGraph(2),
                               direct_mode = false) do node, stage
        @variable(node, 0 <= x <= 1)
        Kokako.parameterize(node, [1, 2, 3], [0.4, 0.5, 0.1]) do ω
            JuMP.set_upper_bound(x, ω)
        end
    end
    node = model[2]
    @test length(node.noise_terms) == 3
    @test JuMP.upper_bound(node.subproblem[:x]) == 1
    node.parameterize(node.noise_terms[2].term)
    @test JuMP.upper_bound(node.subproblem[:x]) == 2
    node.parameterize(3)
    @test JuMP.upper_bound(node.subproblem[:x]) == 3
end

@testset "Kokako.set_stage_objective" begin
    @testset ":Min" begin
        model = Kokako.PolicyGraph(Kokako.LinearGraph(2),
                                   direct_mode = false) do node, stage
            @variable(node, 0 <= x <= 1)
            @stageobjective(node, 2x)
        end
        node = model[2]
        @test node.stage_objective == 2 * node.subproblem[:x]
        @test model.objective_sense == Kokako.MOI.MinSense
    end

    @testset ":Max" begin
        model = Kokako.PolicyGraph(Kokako.LinearGraph(2),
                                   sense = :Max,
                                   direct_mode = false) do node, stage
            @variable(node, 0 <= x <= 1)
            @stageobjective(node, 2x)
        end
        node = model[2]
        @test node.stage_objective == 2 * node.subproblem[:x]
        @test model.objective_sense == Kokako.MOI.MaxSense
    end
end

@testset "Belief Updater" begin
    graph = Kokako.LinearGraph(2)
    Kokako.add_edge(graph, 2 => 1, 0.9)
    model = Kokako.PolicyGraph(graph,direct_mode=false) do subproblem, node
        beliefs = [[0.2, 0.8], [0.7, 0.3]]
        Kokako.parameterize(subproblem, [:A, :B], beliefs[node]) do ω
            return nothing
        end
    end
    belief_updater = Kokako.construct_belief_update(model, [Set([1]), Set([2])])
    belief = Dict(1 => 1.0, 2 => 0.0)
    @test belief_updater(belief, 2, :A) == Dict(1 => 0.0, 2 => 1.0)
    belief = Dict(1 => 0.0, 2 => 1.0)
    @test belief_updater(belief, 1, :B) == Dict(1 => 1.0, 2 => 0.0)


    belief_updater = Kokako.construct_belief_update(model, [Set([1, 2])])

    belief = Dict(1 => 1.0, 2 => 0.0)
    @test belief_updater(belief, 1, :A) == Dict(1 => 0.0, 2 => 1.0)
    belief = Dict(1 => 0.0, 2 => 1.0)
    @test belief_updater(belief, 1, :B) == Dict(1 => 1.0, 2 => 0.0)

    function is_approx(x::Dict{T, Float64}, y::Dict{T, Float64}) where T
        if length(x) != length(y)
            return false
        end
        for (key, value) in x
            if !(value ≈ y[key])
                return false
            end
        end
        return true
    end
    belief = Dict(1 => 0.6, 2 => 0.4)
    @test is_approx(
        belief_updater(belief, 1, :A),
        Dict(1 => 6 / 41, 2 => 35 / 41)
    )
end
