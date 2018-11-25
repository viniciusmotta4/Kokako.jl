using Kokako, GLPK, Test

function infinite_belief()
    graph = Kokako.Graph(
        :root_node,
        [:A, :B],
        [
            (:root_node => :A, 0.5), (:root_node => :B, 0.5),
            (:A => :A, 0.9), (:B => :B, 0.9)
        ],
        belief_partition = [ [:A, :B] ]
    )

    model = Kokako.PolicyGraph(graph,
                bellman_function = Kokako.AverageCut(lower_bound = 0.0),
                optimizer = with_optimizer(GLPK.Optimizer)
                    ) do subproblem, node
        @variables(subproblem, begin
            0 <= inventory <= 3, (Kokako.State, initial_value = 0.0)
            units_bought >= 0
            lost_demand >= 0
            destroyed_units >= 0
            demand
        end)
        @constraint(subproblem, inventory.out == inventory.in - demand +
            units_bought + lost_demand - destroyed_units)
        @stageobjective(subproblem,
            units_bought + 2 * destroyed_units + 10 * lost_demand)
        probabilities = Dict(
            :A => [0.5, 0.3, 0.2],
            :B => [0.2, 0.3, 0.5]
        )
        Kokako.parameterize(subproblem, [1, 2, 3], probabilities[node]) do ω
            JuMP.fix(demand, ω)
        end
    end
end

model = infinite_belief()
Kokako.train(model; iteration_limit = 40, print_level = 2)
