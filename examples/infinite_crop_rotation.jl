#  Copyright 2018, Oscar Dowson.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

using Kokako, GLPK, Test

#=
    We have three growing seasons (stages): soy -> corn -> beef -> soy.

    Let's assume that the yield benefits are something like this:

                |   Current   |
                | soy    corn |
    --------------------------|
          soy   | 0.8    1.2  |
    Prev  corn  | 1.2    0.8  |
    --------------------------+

    What are the states?
    To account for harvestable production, we need one state for each square in
    the yield table:
    - Soy  -> Soy
    - Soy  -> Corn
    - Corn -> Soy
    - Corn -> Corn
    We also need to keep track of the land that was last in soy or corn to
    account for years in which we do not use 100% of the area.
    - Land last in soy
    - Land last in corn
    Since our decisions are decision-hazard, we also need:
    - Number of cattle on the farm

    What are the controls?
    - The amount of land to plant in soy
    - The amount of land to plant in corn
    - The number of cattle on the farm

    What is the uncertainty and how is it revealed?
    - At the start of the soy season, the farmer learns the price of beef and
      the cost of planting soy.
    - At the start of the corn season, the farmer learns the price and yield of
      soy, and the cost of planting corn.
    - At the start of the beef season, the farmer learns the price and yield of
      their corn crop (and grass underneath), and the cost of purchasing cattle.
=#
function infinite_crop_rotation()
    graph = Kokako.Graph(
        :root_node,
        [:soy, :corn, :beef],
        [
            (:root_node => :soy, 1.0),
            (:soy => :corn, 1.0),
            (:corn => :beef, 1.0),
            (:beef => :soy, 0.9)
        ]
    )
    yield_multiplier = JuMP.JuMPArray(
        [0.8, 1.2, 1.0; 1.2, 0.8, 1.0], [:soy, :corn], [:soy, :corn, :none])

    AREA = 1.0  # hectares.

    model = Kokako.PolicyGraph(graph,
                bellman_function = Kokako.AverageCut(lower_bound = 0),
                optimizer = with_optimizer(GLPK.Optimizer)
                    ) do subproblem, node
        crops = [:soy, :corn]
        @variable(subproblem, planted_area[crops, crops], Kokako.State,
            initial_value = 0.0)
        @variable(subproblem, land_state[crops], Kokako.State,
            initial_value = 0.0)

        # A "land-use" balance constraint. This tallies the area last used for
        # each type of crop.
        @constraint(subproblem, [c = crops],
            land_state[c].out == land_state[c].in +
                sum(plant[c2, c].in for c2 in crops) -
                sum(plant[c, c2].out for c2 in crops)
        )

        # We can only plant `cnew` over the old crop `c` if it was already
        # there.
        @constraint(subproblem, [c = crops]
            sum(planted_area[c, cnew].out for cnew in crops) <= land_state[c].in
        )

        # Capacity constraint.
        @constraint(subproblem, sum(land_state[c].out for c in crops) <= AREA)

        @variable(subproblem yield[crops] >= 0)
        # Add some calculations for yield so that we can modify it later in
        # Kokako.paramterize. We put a dummy `1.0` here, but don't worry it will
        # be modified later to be the actual yield.
        @constraint(subproblem, yield_constraints[current_crop in crops],
            yield[current_crop] == sum(
                1.0 * planted_area[previous_crop, current_crop].in
                for previous_crop in crops)
        )

        # Deal with the cattle.
        @variable(num_cattle >= 0, Kokako.State, initial_value = 0)
        @variable(subproblem, cattle_purchases >= 0)
        @variable(subproblem, cattle_sales >= 0)
        @constraint(subproblem, num_cattle.out ==
            num_cattle.in + cattle_purchases - cattle_sales)

        Kokako.parameterize(subproblem, [1, 2]) do ω
            # Note: we have to be careful here; JuMP will normalize the
            # constraint by shifting all of the terms the the LHS, so we need to
            # set a -ve yield coefficient.
            for previous_crop in crops
                for current_crop in crops
                    ρ = yield_multiplier[previous_crop, current_crop]
                    JuMP.set_coefficient(
                        yield_constraint[current_crop],
                        planted_area[previous_crop, current_crop].in,
                        -ρ * ω
                    )
                end
            end
        end
        @stageobjective(subproblem, 2.0)
    end
end
