import json
import os
import matplotlib.pyplot as plt

# Assuming you have a directory path. Replace 'your_directory_path' with the actual path.
# directory_path = 'your_directory_path'


def sum_success_counters_and_file_statistics(directory_path):
    total_success_counter = 0
    json_file_count = 0  # Total number of JSON files
    success_counter_statistics = {}  # Dictionary to store count of files for each success_counter value

    # List all files in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith('.json'):
            filepath = os.path.join(directory_path, filename)
            # Open and read the JSON file
            with open(filepath, 'r') as file:
                data = json.load(file)
                # Sum up the success_counter values
                success_counter = data.get('success_counter', 0)
                total_success_counter += success_counter
                json_file_count += 1  # Increment the counter for each JSON file processed

                # Update statistics for success_counter
                if success_counter in success_counter_statistics:
                    success_counter_statistics[success_counter] += 1
                else:
                    success_counter_statistics[success_counter] = 1

    return total_success_counter, json_file_count, success_counter_statistics


# Call the function and get the results
total, count, statistics = sum_success_counters_and_file_statistics("/Users/username/Desktop/TUM/Courses/Sem 3/DL_Robotics/Project/Grasping_Ours/evals")

# Plotting success_counter vs. file_count
success_counters = list(statistics.keys())
file_counts = [statistics[sc] for sc in success_counters]

print(f'total:{total}, count:{count}')
plt.figure(figsize=(10, 6))
plt.bar(success_counters, file_counts, color='skyblue')
plt.xlabel('Number of successful grasps')
plt.ylabel('Number of object')
plt.title('Quantitative Result on Validation dataset')
plt.xticks(success_counters)
plt.show()