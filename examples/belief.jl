using Kokako, GLPK, Random, Statistics, Test

function infinite_belief()
    graph = Kokako.Graph(
        :root_node,
        [:Ad, :Ah, :Bd, :Bh],
        [
            (:root_node => :Ad, 0.5), (:root_node => :Bd, 0.5),
            (:Ad => :Ah, 1.0), (:Ah => :Ad, 0.9),
            (:Bd => :Bh, 1.0), (:Bh => :Bd, 0.9)
        ],
        belief_partition = [ [:Ad, :Bd], [:Ah, :Bh] ]
    )
    model = Kokako.PolicyGraph(graph,
                bellman_function = Kokako.AverageCut(lower_bound = 0.0),
                optimizer = with_optimizer(GLPK.Optimizer)
                    ) do subproblem, node
        @variables(subproblem, begin
            0 <= inventory <= 3, (Kokako.State, initial_value = 0.0)
            0 <= units_bought <= 3, (Kokako.State, initial_value = 0.0)
            lost_demand >= 0
            destroyed_units >= 0
            demand
        end)
        if node == :Ad || node == :Bd
            @constraint(subproblem, inventory.out == inventory.in)
            @stageobjective(subproblem, units_bought.out)
        else
            @constraint(subproblem, inventory.out == inventory.in +
                units_bought.in - demand + lost_demand - destroyed_units
            )
            @stageobjective(subproblem, 2 * destroyed_units + 10 * lost_demand)
            probabilities = Dict(
                :Ah => [0.5, 0.3, 0.2],
                :Bh => [0.2, 0.3, 0.5]
            )
            Kokako.parameterize(subproblem, [1, 2, 3], probabilities[node]) do ω
                JuMP.fix(demand, ω)
            end
        end
    end
end

Random.seed!(123)

model = infinite_belief()
Kokako.train(model; iteration_limit = 200, print_level = 1)

simulations = Kokako.simulate(model, 500)
objectives = map(sim -> sum(s[:stage_objective] for s in sim), simulations)

sample_mean = round(Statistics.mean(objectives); digits = 2)
sample_ci = round(1.96 * Statistics.std(objectives) / sqrt(500); digits = 2)
println("Confidence_interval = $(sample_mean) ± $(sample_ci)")

@test sample_mean - sample_ci <=
    Kokako.calculate_bound(model) <=
    sample_mean + sample_ci

if length(ARGS) > 0
    simulations = Kokako.simulate(model, 500,
        [:units_bought, :demand, :inventory, :lost_demand, :destroyed_units];
        sampling_scheme = Kokako.InSampleMonteCarlo(
            max_depth = 20,
            terminate_on_dummy_leaf = false
        )
    )

    plt = Kokako.spaghetti_plot(stages = 20, scenarios = 500)
    Kokako.add_spaghetti(plt; title = "Actual") do scenario, stage
        simulations[scenario][stage][:node_index] == :A ? 1.0 : 0.0
    end
    Kokako.add_spaghetti(plt; title = "Belief") do scenario, stage
        simulations[scenario][stage][:belief][:A]
    end
    Kokako.add_spaghetti(plt; title = "Inventory") do scenario, stage
        simulations[scenario][stage][:inventory].out
    end
    Kokako.add_spaghetti(plt; title = "Demand") do scenario, stage
        simulations[scenario][stage][:demand]
    end
    Kokako.add_spaghetti(plt; title = "Units bought") do scenario, stage
        simulations[scenario][stage][:units_bought]
    end
    Kokako.add_spaghetti(plt; title = "Lost demand") do scenario, stage
        simulations[scenario][stage][:lost_demand]
    end
    Kokako.add_spaghetti(plt; title = "Destroyed units") do scenario, stage
        simulations[scenario][stage][:destroyed_units]
    end
    Base.show(plt)
end
