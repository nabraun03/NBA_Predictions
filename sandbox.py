import math

# Initialize jobs for PS scheduling
ps_jobs = [
    {"name": "A", "arrival": 0, "size": 3, "remaining": 3, "completion": 0},
    {"name": "B", "arrival": 1, "size": 2, "remaining": 2, "completion": 0},
    {"name": "C", "arrival": 3, "size": 3, "remaining": 3, "completion": 0},
    {"name": "D", "arrival": 7, "size": 4, "remaining": 4, "completion": 0},
    {"name": "E", "arrival": 10, "size": 5, "remaining": 5, "completion": 0},
    {"name": "F", "arrival": 12, "size": 3, "remaining": 3, "completion": 0}
]

current_time_ps = 0
while any(job['remaining'] > 0 for job in ps_jobs):
    # Filter jobs that have arrived by current_time
    active_jobs = [job for job in ps_jobs if job['arrival'] <= current_time_ps and job['remaining'] > 0]
    if not active_jobs:
        # If there are no active jobs, advance time to the arrival of the next job
        current_time_ps = min(job['arrival'] for job in ps_jobs if job['remaining'] > 0)
        continue

    # Divide the processor's time equally among all active jobs
    time_share = 1 / len(active_jobs)
    for job in active_jobs:
        # Update the remaining time for each job based on the time share
        job['remaining'] -= time_share
        # If a job completes, update its completion time
        if job['remaining'] <= 0 and job['completion'] == 0:
            job['completion'] = current_time_ps + time_share * (1 + job['remaining'])
    
    # Advance the current time
    current_time_ps += time_share
    print(ps_jobs)

# Round completion times to the nearest integer, with 0.5 rounding up
ps_completion_times = {job['name']: job['completion'] for job in ps_jobs}
print(ps_completion_times)
