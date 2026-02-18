import torch
import copy

def compute_router_order(model, num_layers):
    routers_order = []
    routers_values = []

    state_dict = model.state_dict()

    for i in range(num_layers):
        weight = state_dict[
            f"model.layers.{i}.block_sparse_moe.gate.weight"
        ]

        values, indices = torch.sort(
            torch.linalg.vector_norm(weight, ord=2, dim=1)
        )

        routers_order.append(indices.tolist())
        routers_values.append(values.tolist())
    return routers_order, routers_values

def compute_variance_scores(model, num_layers, num_experts):
    t_1 = []
    state_dict = model.state_dict()

    for layer in range(num_layers):
        layer_scores = []

        for expert in range(num_experts):
            weight = state_dict[
                f"model.layers.{layer}.block_sparse_moe.experts.{expert}.w1.weight"
            ]

            var = torch.var(weight, dim=1)

            norms, idx = torch.sort(
                torch.linalg.vector_norm(weight, ord=2, dim=1)
            )

            # Variance of highest-norm neuron
            layer_scores.append(var[idx[-1]].item())

        t_1.append(layer_scores)

    return t_1

def compute_combined_order(
    routers_order,
    variance_scores,
    zeta: float,
):
    experts_order = []

    num_layers = len(routers_order)

    for layer in range(num_layers):

        c_l = copy.deepcopy(routers_order[layer])

        for expert_id in reversed(routers_order[layer]):

            temp = [
                k
                for k, val in enumerate(
                    [variance_scores[layer][l] for l in c_l]
                )
                if (zeta * val)
                < variance_scores[layer][expert_id]
            ]

            if temp:
                if c_l.index(expert_id) > temp[0]:
                    y = c_l.pop(c_l.index(expert_id))
                    c_l.insert(temp[0], y)

        experts_order.append(c_l)

    return experts_order

def compute_variance_order(variance_scores):
    experts_order = []

    for layer_scores in variance_scores:
        order = sorted(
            range(len(layer_scores)),
            key=lambda x: layer_scores[x]
        )
        experts_order.append(order)

    return experts_order


def validate_experts_order(experts_order, num_experts=8):

    for layer_id, layer in enumerate(experts_order):
        if len(layer) != num_experts:
            raise ValueError(
                f"Layer {layer_id} does not have {num_experts} experts."
            )


def assign_bits_two_level(
    experts_order,
    bh: int,
    bl: int,
    b_avg: float,
):

    num_layers = len(experts_order)
    num_experts = len(experts_order[0])

    validate_experts_order(experts_order, num_experts)

    kappa = (b_avg - bl) / (bh - bl)
    kappa = max(0.0, min(1.0, kappa))

    n_high = int(round(kappa * num_experts))

    bit_dict = {}

    for layer_id, order in enumerate(experts_order):

        layer_bits = {}

        for rank, expert_id in enumerate(order):

            if rank < n_high:
                layer_bits[expert_id] = bh
            else:
                layer_bits[expert_id] = bl

        bit_dict[layer_id] = layer_bits

    return bit_dict

def solve_three_level_counts_policy(
    num_experts: int,
    b_avg: float,
    bh: int,
    bm: int,
    bl: int,
):
    """
    Three-level bit allocation policy.

    Procedure:

    1. Check equal spacing of bit levels.
    2. Compute two-level baseline:
         • If b_avg >= bm → use (bh, bm)
         • If b_avg <= bm → use (bm, bl) with bh = 0
    3. Reallocate experts:
         bm → bh
         bm → bl
       keeping same average.
    4. Stop based on region constraints.

    Returns:
        (n_l, n_m, n_h)
    """

    if (bh - bm) != (bm - bl):
        raise ValueError(
            "For three-bit level, the three-bit levels should be equally spaced."
        )

    if b_avg >= bm:
        frac = (b_avg - bm) / (bh - bm)
        frac = max(0.0, min(1.0, frac))

        n_h = int(round(frac * num_experts))
        n_m = num_experts - n_h
        n_l = 0

        base_n_h = n_h

    else:
        frac = (b_avg - bl) / (bm - bl)
        frac = max(0.0, min(1.0, frac))

        n_m = int(round(frac * num_experts))
        n_l = num_experts - n_m
        n_h = 0

        base_n_h = 0

    delta = bh - bl

    r_high = bh - delta / 3
    r_mid = bh - 2 * delta / 3

    if b_avg > r_high:
        region = "bh"

    elif b_avg >= r_mid:
        region = "bm"

    else:
        region = "bl"

    while True:

        if n_m < 2:
            break

        cand_n_l = n_l + 1
        cand_n_m = n_m - 2
        cand_n_h = n_h + 1

        if region == "bh":
            pass

        elif region == "bm":
            promoted = cand_n_h - base_n_h

            if cand_n_m < promoted:
                break

        else:
            if cand_n_l > n_l:
                break

        n_l = cand_n_l
        n_m = cand_n_m
        n_h = cand_n_h

    return n_l, n_m, n_h

def assign_bits_three_level(
    experts_order,
    bh: int,
    bm: int,
    bl: int,
    b_avg: float,
):

    num_layers = len(experts_order)
    num_experts = len(experts_order[0])

    validate_experts_order(experts_order, num_experts)

    n_l, n_m, n_h = solve_three_level_counts_policy(
        num_experts,
        b_avg,
        bh,
        bm,
        bl,
    )

    bit_dict = {}

    for layer_id, order in enumerate(experts_order):

        layer_bits = {}

        for rank, expert_id in enumerate(order):

            if rank < n_h:
                layer_bits[expert_id] = bh

            elif rank < n_h + n_m:
                layer_bits[expert_id] = bm

            else:
                layer_bits[expert_id] = bl

        bit_dict[layer_id] = layer_bits

    return bit_dict

def create_bit_distribution(
    experts_order,
    b_avg: float,
    bh: int,
    bm: int = None,
    bl: int = 1,
):
    if bm is None:

        return assign_bits_two_level(
            experts_order,
            bh,
            bl,
            b_avg,
        )

    else:

        return assign_bits_three_level(
            experts_order,
            bh,
            bm,
            bl,
            b_avg,
        )