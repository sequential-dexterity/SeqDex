


def process_dapg(args, env, cfg_train, logdir):
    from algorithms.il.dapg import DAPG, ActorCritic, ActorCriticPointCloud
    learn_cfg = cfg_train["learn"]
    is_testing = learn_cfg["test"]
    # is_testing = True
    # Override resume and testing flags if they are passed as parameters.
    if args.model_dir != "":
        is_testing = True
        chkpt_path = args.model_dir

    logdir = logdir + "_seed{}".format(env.task.cfg["seed"])

    if cfg_train["point_cloud"]["use_point_cloud"] == True:
        actor_critic = ActorCriticPointCloud
    else:
        actor_critic = ActorCritic


    """Set up the dapg system for training or inferencing."""
    dapg = DAPG(vec_env=env,
              actor_critic_class=actor_critic,
              num_transitions_per_env=learn_cfg["nsteps"],
              num_learning_epochs=learn_cfg["noptepochs"],
              num_mini_batches=learn_cfg["nminibatches"],
              clip_param=learn_cfg["cliprange"],
              gamma=learn_cfg["gamma"],
              lam=learn_cfg["lam"],
              init_noise_std=learn_cfg.get("init_noise_std", 0.3),
              value_loss_coef=learn_cfg.get("value_loss_coef", 2.0),
              entropy_coef=learn_cfg["ent_coef"],
              learning_rate=learn_cfg["optim_stepsize"],
              max_grad_norm=learn_cfg.get("max_grad_norm", 2.0),
              use_clipped_value_loss=learn_cfg.get("use_clipped_value_loss", False),
              schedule=learn_cfg.get("schedule", "fixed"),
              desired_kl=learn_cfg.get("desired_kl", None),
              model_cfg=cfg_train["policy"],
              device=env.rl_device,
              sampler=learn_cfg.get("sampler", 'sequential'),
              log_dir=logdir,
              is_testing=is_testing,
              print_log=learn_cfg["print_log"],
              apply_reset=False,
              asymmetric=(env.num_states > 0)
              )

    # dapg.test("./logs/allegro_hand_catch_underarm/dapg/dapg_seed-1/model_6000.pt")
    if cfg_train["collect_data"] and args.model_dir != "":
        print("Loading model from {}".format(chkpt_path))
        dapg.load_collection(chkpt_path)
    elif is_testing and args.model_dir != "":
        print("Loading model from {}".format(chkpt_path))
        dapg.test(chkpt_path)
    elif args.model_dir != "":
        print("Loading model from {}".format(chkpt_path))
        dapg.load(chkpt_path)
    

    return dapg

