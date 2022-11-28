NUM_USER_DATA_PER_ACT = 1
NUM_ACTUATOR_DATA = 1


cdef struct antagonistic_params:
    mjtNum moment_1
    mjtNum lce_1_ref
    mjtNum moment_2
    mjtNum lce_2_ref


cdef struct antagonistic_lengths:
    mjtNum lce_1
    mjtNum lce_2


cdef antagonistic_params compute_parametrization(mjtNum phi_min, mjtNum phi_max, mjtNum lce_min, mjtNum lce_max):
    """
    Find parameters for muscle length computation.
    This should really only be done once...
    """
    cdef mjtNum eps = 0.001
    cdef mjtNum moment_1 = ((lce_max - lce_min + eps) / (phi_max - phi_min + eps))
    cdef mjtNum lce_1_ref = (lce_min - moment_1 * phi_min)
    cdef mjtNum moment_2 = ((lce_max - lce_min + eps) / (phi_min - phi_max + eps))
    cdef mjtNum lce_2_ref = (lce_min - moment_2 * phi_max)
    return antagonistic_params(moment_1, lce_1_ref, moment_2, lce_2_ref)


cdef antagonistic_lengths compute_virtual_lengths(const mjModel *m, const mjData *d, antagonistic_params* params, int id):
    """
    Compute muscle fiber lengths l_ce depending on joint angle
    Attention: The mapping of actuator_trnid to qvel is only 1:1 because we have only 
    slide or hinge joints and no free joint!!! Otherwise you have to build this mapping
    by looking at every single joint type.
    """
    lce_1 = d.qpos[m.actuator_trnid[id * 2]] * params.moment_1 + params.lce_1_ref
    lce_2 = d.qpos[m.actuator_trnid[id * 2]] * params.moment_2 + params.lce_2_ref
    return antagonistic_lengths(lce_1, lce_2)


cdef mjtNum compute_moment(const antagonistic_params* params, antagonistic_lengths* lengths, const mjModel *m, const mjData *d, int id):
    """
    Joint moments are computed from muscle contractions and then returned
    """
    act1 = d.userdata[4 + 2 * id]
    act2 = d.userdata[4 + 2 * id + 1]
    cdef mjtNum vmax = m.actuator_gainprm[id * mjNGAIN + 6]
    cdef mjtNum lce_dot_1 = get_vel(m, d, params.moment_1, vmax, id, 0)
    cdef mjtNum lce_dot_2 = get_vel(m, d, params.moment_2, vmax, id, 1)
    cdef mjtNum F1 = FL(m, lengths.lce_1, id) * FV(m, lce_dot_1, id) * act1 + FP(m, lengths.lce_1, id)
    cdef mjtNum F2 = FL(m, lengths.lce_2, id) * FV(m, lce_dot_2, id) * act2 + FP(m, lengths.lce_2, id)
    F1 = F1 *  get_peak_force(m, id)
    F2 = F2 *  get_peak_force(m, id)
    write_lce_data_to_userarray(lce_dot_1, lce_dot_2, F1, F2, m, d, id)
    # Return minus because of mujoco actuator definition, the muscle pulls
    return -(F1 * params.moment_1 + F2 * params.moment_2)


cdef write_lce_data_to_userarray(mjtNum lce_dot_1, mjtNum lce_dot_2, mjtNum F1, mjtNum F2, const mjModel *m, const mjData *d, int id):
    d.userdata[m.nuserdata - 4 * m.nu + 2 * id] = F1
    d.userdata[m.nuserdata - 4 * m.nu + 2 * id + 1] = F2 
    d.userdata[m.nuserdata - 6 * m.nu + 2 * id] = lce_dot_1
    d.userdata[m.nuserdata - 6 * m.nu + 2 * id + 1] = lce_dot_2
        

cdef apply_moment(const mjData *d, mjtNum moment, int id):
    """
    Not used atm, could apply forces directly instead of returning
    moment to actuator.
    """
    d.qfrc_applied[id] = moment


cdef mjtNum get_vel(const mjModel *m, const mjData *d, mjtNum moment, mjtNum vmax, int id, int muscle_id):
    """
    For muscle 1, the joint angle increases if it pulls. This means 
    that the joint and the muscle velocity have opposite signs. But this is already
    included in the value of the moment arm. So we don't need if/else branches here.
    Attention: The mapping of actuator_trnid to qvel is only 1:1 because we have only 
    slide or hinge joints and no free joint!!! Otherwise you have to build this mapping
    by looking at every single joint type.
    """
    return moment * d.qvel[m.actuator_trnid[id * 2]]


cdef mjtNum FL(const mjModel *m, mjtNum lce, int id):
    """
    Force length
    """
    # these correspond to the lmin lmax in the gainprm,
    # not the ones used in the paper
    cdef mjtNum lmin = m.actuator_gainprm[id * mjNGAIN + 4]
    cdef mjtNum lmax = m.actuator_gainprm[id * mjNGAIN + 5]
    #return 1.0
    return bump(lce, lmin, 1, lmax) + 0.15 * bump(lce, lmin, 0.5 * (lmin + 0.95), 0.95)


cdef mjtNum FV(const mjModel *m, mjtNum lce_dot, int id):
    """
    Force velocity
    """
    cdef mjtNum vmax = m.actuator_gainprm[id * mjNGAIN + 6]
    cdef mjtNum fvmax = m.actuator_gainprm[id * mjNGAIN + 8]
    cdef mjtNum c = fvmax - 1
    return force_vel(lce_dot, c, vmax, fvmax)


cdef mjtNum FP(const mjModel *m, mjtNum lce, int id):
    """
    Force passive
    """
    cdef mjtNum lmax = m.actuator_gainprm[id * mjNGAIN + 5]
    cdef mjtNum b = 0.5 * (lmax  + 1)
    cdef mjtNum fpmax = m.actuator_gainprm[id * mjNGAIN + 7]
    #return 0.0
    return passive_force(lce, fpmax, b)


cdef mjtNum get_peak_force(const mjModel *m, int id):
    cdef mjtNum force = m.actuator_gainprm[id * mjNGAIN + 2]
    cdef mjtNum scale = m.actuator_gainprm[id * mjNGAIN + 3]
    cdef mjtNum actuator_acc0 = m.actuator_acc0[id]
    if (force + 1) < 0.01:
        return scale / actuator_acc0
    else:
        return force


cdef mjtNum bump(mjtNum length, mjtNum A, mjtNum mid, mjtNum B):
    """
    Force length relationship as implemented by MuJoCo.
    """
    cdef mjtNum left = 0.5 * (A + mid)
    cdef mjtNum right = 0.5 * (mid + B)
    cdef mjtNum temp = 0

    if ((length <= A) or (length >= B)):
        return 0
    elif (length < left):
        temp = (length - A) / (left - A)
        return 0.5 * temp * temp
    elif (length < mid):
        temp = (mid - length) / (mid - left)
        return 1 - 0.5 * temp * temp
    elif (length < right):
        temp = (length - mid) / (right - mid)
        return 1 - 0.5 * temp * temp
    else:
        temp = (B - length) / (B - right)
        return 0.5 * temp * temp


cdef mjtNum passive_force(mjtNum length, mjtNum fpmax, mjtNum b):
    """Parallel elasticity (passive muscle force) as implemented
    by MuJoCo.
    """
    cdef mjtNum temp = 0

    if (length <= 1):
        return  0
    elif (length <= b):
        temp = (length -1) / (b - 1)
        return 0.25 * fpmax * temp * temp * temp
    else:
        temp = (length - b) / (b - 1)
        return 0.25 * fpmax * (1 + 3 * temp)


cdef mjtNum force_vel(mjtNum velocity, mjtNum c, mjtNum vmax, mjtNum fvmax):
    """
    Force velocity relationship as implemented by MuJoCo.
    """
    cdef mjtNum eff_vel = velocity / vmax
    #print('eff vel is')
    #print(eff_vel)
    if (eff_vel < -1):
        return 0
    elif (eff_vel <= 0):
        return (eff_vel + 1) * (eff_vel + 1)
    elif (eff_vel <= c):
        return fvmax - (c - eff_vel) * (c - eff_vel) / c
    else:
        return fvmax


cdef mjtNum c_zero_gains(const mjModel*m, const mjData*d, int id) with gil:
    '''Actuator is implemented as a bias function because it sidesteps the issue
    of a linear controller. Normally the gain output is multiplied with the 
    ctrl signal, we do this manually in the bias now.'''
    return 0.0


cdef mjtNum activ_dyn(const mjModel* m, const mjData* d, int id):
    """
    Activity and controls have to be written inside userdata. Assume
    two virtual muscles per real mujoco actuator and let's roll.
    """
    act1 = d.userdata[4 + 2 * id]
    act2 = d.userdata[4 + 2 * id + 1]
    ctrl1 = d.userdata[4 + 2 * m.nu + 2 * id ]
    ctrl2 = d.userdata[4 + 2 * m.nu + 2 * id + 1]
    d.userdata[4 + 2 * id] = 100.0 * (ctrl1 - act1) * m.opt.timestep + act1
    d.userdata[4 + 2 * id + 1] = 100.0 * (ctrl2 - act2) * m.opt.timestep + act2
    return 0


cdef mjtNum geometry_free_muscle_bias(const mjModel *m, const mjData *d, int id):
    """
    Here the muscle forces are computed. We just provide a bias as gains undergo
    additional MuJoCo transformations. The user parameters below are the only ones we 
    have in addition to the ones in the MuJoCo general actuator.
    """
    # ----------------user parameters ------------------------
    # allowed joint angles
    #cdef mjtNum phi_min = -0.8 * 3.14
    #cdef mjtNum phi_max = 0.8 * 3.14
    cdef mjtNum phi_min = -0.5 * 3.14
    cdef mjtNum phi_max = 0.5 * 3.14

    # corresponding min/max muscle lengths normalized by the optimal isometric length
    # this corresponds to l_min/l_max in the paper, and range_0 range_1 in the mujoco
    # gainprm
    cdef mjtNum lce_min = 0.75
    cdef mjtNum lce_max = 1.05
    # ----------------user parameters ------------------------
    
    cdef antagonistic_params params = compute_parametrization(phi_min, phi_max, lce_min, lce_max)
    cdef antagonistic_lengths lengths = compute_virtual_lengths(m, d, &params, id)
    d.userdata[m.nuserdata - (2 * m.nu) + 2 * id] = lengths.lce_1
    d.userdata[m.nuserdata - (2 * m.nu) + 2 * id + 1] = lengths.lce_2
    cdef mjtNum moment = compute_moment(&params, &lengths, m, d, id)
    return moment


def set_muscle_control(m, d):
    """
    This function should be called by your python as cymj.set_muscle_control(env.model, env.data)
    """
    global mjcb_act_gain
    global mjcb_act_bias
    global mjcb_act_dyn
    for act_idx in range(m.nu):
        if m.jnt_type[act_idx] == 0 or m.jnt_type[act_idx] == 1:
            raise Exception('The current retrieval of joint position and velocity for the muscle model is only \
            applicable for slide and hinge joints!')
    if m.nuserdata < (m.nu * NUM_ACTUATOR_DATA + NUM_ACTUATOR_DATA):
        raise Exception(f'nuserdata is not set large enough to store PID internal states. It is {m.nuserdata} but should be {m.nu * NUM_USER_DATA_PER_ACT}')

    if m.nuser_actuator < m.nu * NUM_USER_DATA_PER_ACT:
        raise Exception(f'nuser_actuator is not set large enough to store controller types. It is {m.nuser_actuator} but should be {m.nu * NUM_ACTUATOR_DATA}')

    for i in range(m.nuserdata):
        d.userdata[i] = 0.0
    mjcb_act_dyn = activ_dyn
    mjcb_act_gain = c_zero_gains
    mjcb_act_bias = geometry_free_muscle_bias
