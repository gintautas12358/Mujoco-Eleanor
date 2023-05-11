

def simEpisode(env, max_ep_len, action_func=None, enableStore=False, replay_buffer=None):
    if action_func is None:
        raise Exception("no action function given as a parameter for simEpisode.")
    
    o, d, ep_ret, ep_len = env.reset(), False, 0, 0
    o = o[0]
    while not(d or (ep_len == max_ep_len)):
        # Take deterministic actions at test time 
        a = action_func(o)
        a = a.flatten()
        o2, r, d, _, info = env.step(a)
        if enableStore:
            if replay_buffer is None:
                raise Exception("if enableStore is true, replay_buffer is also needed for simEpisode.")
            replay_buffer.store(o, a, r, o2, d)
        o = o2
        ep_ret += r
        ep_len += 1
    
    return ep_ret, ep_len
