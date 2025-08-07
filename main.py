from QKP.Instances.run_experiments import run_multiple_experiments

if __name__ == "__main__":

        n_items = 5                    # Number of items
        r = 50                         # range of coefficients
        pcts = [25, 50, 75, 100]       # Instance density
        instance_type = "standard"     # Type of instance to generate
        
        for pct in pcts:
                run_multiple_experiments(n_items=n_items, r=r, pct=pct, n_experiments=20, instance_type=instance_type)
