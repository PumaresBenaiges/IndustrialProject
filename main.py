import sys
from class_sample_testing import *

def plot_ctq_values(values, title, threshold, sample_labels, axis_name):
    x = np.arange(len(values[0]))
    width = 0.12

    plt.figure(figsize=(12, 6))
    plt.bar(x - 2.5 * width, values[0], width, label="Trial 1 Operator 1")
    plt.bar(x - 1.5 * width, values[1], width, label="Trial 1 Operator 2")
    plt.bar(x - 0.5 * width, values[2], width, label="Trial 1 Operator 3")
    plt.bar(x + 0.5 * width, values[3], width, label="Trial 2 Operator 1")
    plt.bar(x + 1.5 * width, values[4], width, label="Trial 2 Operator 2")
    plt.bar(x + 2.5 * width, values[5], width, label="Trial 2 Operator 3")
    plt.axhline(y=threshold, color="red", linestyle="--", linewidth=1.5, label="Threshold: "+str(threshold))
    plt.xticks(x, sample_labels)
    plt.xlabel("Samples", fontsize=16)
    #plt.ylabel("Values")
    plt.ylabel(axis_name,  fontsize=16)
    #plt.title(title)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.show()

def plot_ctq_values_train(vals, labels, threshold, title, axis_name):
    plt.figure(figsize=(10, 6))
    plt.bar(labels, vals, color="skyblue")
    plt.axhline(y=threshold, color="red", linestyle="--", linewidth=1.5, label="Threshold")
    plt.ylabel(axis_name,  fontsize=16)
    #plt.title(title)
    plt.legend()
    plt.xticks(rotation=45)
    plt.yticks(fontsize=16)
    plt.xlabel("Samples", fontsize=16)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    # Read args
    if len(sys.argv) > 3:
        base_folder = sys.argv[1]
    else:
        base_folder = "./IDP Group A"
    n_trials = 2
    n_operators = 3

    # Classify Train samples
    train = sample_testing(folder_name=base_folder+"/train")
    train_result = train.test_sample()

    # Classify Test samples
    trials_results = []
    for trial in range(1,n_trials+1):
        for operator in range(1,n_operators+1):
            folder = base_folder + "/test"+ str(trial)+ "/operator"+ str(operator)
            test = sample_testing(folder_name=folder, vis=False, trial=trial, operator=operator)
            result = test.test_sample()
            trials_results.append(result)

    # Save results to excel file
    train_result.to_excel("train_results.xlsx", index=False)
    combined = pd.concat(trials_results, ignore_index=True)
    combined.to_excel("trials_results.xlsx", index=False)

    # Plot CTQ values for Train samples
    t_deltae = train_result["DE_mean"].values 
    t_radii = train_result["D_radius"].values 
    plot_ctq_values_train(t_deltae, train.sample_folders, 2.8, "Average Color Difference Between Reference and Defects", "DE (CIE2000)")
    plot_ctq_values_train(t_radii, train.sample_folders, 1.2, "Radius Difference Between Reference and Defects", "Radius difference")

    # Plot CTQ values for Test samples
    samples_including_s0 = test.sample_folders
    all_deltae = [t["DE_mean"].values for t in trials_results]
    all_radius = [t["D_radius"].values for t in trials_results]

    all_deltae = [np.insert(x, 0, np.nan) if len(x) < len(samples_including_s0) else x for x in all_deltae ]
    all_radius = [np.insert(x, 0, np.nan) if len(x) < len(samples_including_s0) else x for x in all_radius ]

    plot_ctq_values(
        all_deltae,
        "Delta E difference Between Reference and Samples",
        2.8,
        samples_including_s0,
        "DE (CIE2000)"
    )
    plot_ctq_values(
        all_radius,
        "Radius difference Between Reference and Samples",
        1.2,
        samples_including_s0,
        "Radius difference"
    )

    

