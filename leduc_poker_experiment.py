from ESCHER import *

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    train_device = 'cpu'
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'tmp', 'results', 'leduc',
                             )
    os.makedirs(save_path, exist_ok=True)

    game = pyspiel.load_game("leduc_poker")

    iters = 300
    num_traversals = 500
    num_val_fn_traversals = 500
    regret_train_steps = 200
    val_train_steps = 200
    policy_net_train_steps = 1000
    batch_size_regret = 256
    batch_size_val = 256

    deep_cfr_solver = ESCHERSolver(
        game,
        num_traversals=int(num_traversals),
        num_iterations=iters,
        check_exploitability_every=10,
        compute_exploitability=True,
        regret_network_train_steps=regret_train_steps,
        policy_network_train_steps=policy_net_train_steps,
        batch_size_regret=batch_size_regret,
        value_network_train_steps=val_train_steps,
        batch_size_value=batch_size_val,
        train_device=train_device,
    )
    if True:
        regret, pol_loss, convs, nodes = deep_cfr_solver.solve(save_path_convs=save_path)

    arr = np.load(os.path.join(save_path, '_convs.npy'))
    plt.plot(arr)
    plt.show()

    quit()
