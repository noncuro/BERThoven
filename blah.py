blah = []
for lr in [5e-6, 1e-5, 2e-5, 3e-5]:
    for combine_inputs in ["concat", "one way"]:
        for dropout in [0, 0.2, 0.5]:
            for loss in ["l1", "l2", "l1_smooth"]:
                for mask in [True, False]:
                    blah.append({
                        "scheduler":"linear",
                        "optimizer":"AdamW",
                        "lr":lr,
                        "eps":5e-8,
                        "loss_function":loss,
                        "combine_inputs":combine_inputs,
                        "epochs":3,
                        "dropout":dropout,
                        "masking":mask,
                        "cls":True,
                        "warmup_proportion":0.1
                    })
