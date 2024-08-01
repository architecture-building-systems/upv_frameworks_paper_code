def pv_watts_method(G_eff, T_cell, P_ref, gamma, T_ref=25, G_ref=1000, I_misc=0.1):
    """
    :param G_eff: effective irradiance (W/sqm.)
    :param T_cell: current cell temperature (˚C)
    :param P_ref: peak power (W)
    :param gamma: maximum power temperature coefficient (% loss / ˚C) (typ -0.00485)
    :param T_ref: cell temperature at test conditions (˚C) Default to 25
    :param G_ref: irradiance at test conditions (W/m2) Default to 1000
    :param I_misc: system losses (-) Default to 0.1
    :return: power with system losses 
    """

    if gamma < -0.02:
        gamma = gamma / 100

    if G_eff > 125:
        P_mp = (G_eff / G_ref) * P_ref * (1 + gamma * (T_cell - T_ref))
    else:
        P_mp = ((0.008 * G_eff ** 2) / G_ref) * P_ref * (1 + gamma * (T_cell - T_ref))
    return P_mp * (1 - I_misc)


# def bishop_workflow():
#     # 