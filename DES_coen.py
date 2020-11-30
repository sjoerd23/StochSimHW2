import simpy
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import pandas as pd 

class Queue(object):
	"""
	A class used to represent a queue

	"""
	def __init__(self, env, num_servers, priority):
		"""
		Args
		----------
		env : Environment
			simpy.environment to host the processes
		num_servers : int
			number of servers
		priority : boolean
			prioritize the shortest tasks
		"""
		self.env = env
		self.priority = priority
		if not priority:
			self.server = simpy.Resource(env, num_servers)
		else:
			self.server = simpy.PriorityResource(env, num_servers)

	def task(self, task_time):
		"""
		execute a task
		"""
		yield self.env.timeout(task_time)


def person(env, name, queue, mu, serv_t_dist):
	"""
	a function to represent a person joining the queue and
	performing a task when specified.

	Args:
		name : int
			unique representation of a person
		queue : Queue
			the queue joined
		mu : float
			mean task time

	Appends the waiting time to the global list wt
	"""

	# time of arrival
	arrival = env.now

	# compute the task time
	if serv_t_dist == "M":
		task_time = -np.log(np.random.random()) / 1
	elif serv_t_dist == "D":
		task_time = 1
	elif serv_t_dist == "LT":
		x = np.random.random()
		if x < 0.75:
			task_time = -np.log(np.random.random()) * 0.5
		else:
			task_time = -np.log(np.random.random()) * 2.5
	else:
		raise ValueError("Invalid type of service time distribution")
	# print("{} arrives at queue at {} with task time {:.2f}".format(name, arrival, task_time))

	# handle priority
	if not queue.priority:
		with queue.server.request() as request:
			# request a position at the queue
			yield request

			# request accepted, server entered
			enter = env.now
			# print("{} is served at {:.2f}".format(name, enter))
			yield env.process(queue.task(task_time))

			# waiting time
			wt.append(enter - arrival)
	else:
		with queue.server.request(priority = task_time, preempt=True) as request:
			yield request

			enter = env.now
			# print("{} is served at {:.2f}".format(name, enter))
			yield env.process(queue.task(task_time))

			wt.append(enter - arrival)


def setup(env, num_servers, lamda, mu, priority, max_persons, serv_t_dist):
	"""
	sets up the system

	Args:
		env : Environment
			simpy.environment to host the processes
		num_servers : int
			number of servers
		lamda : float
			mean arrival rate
		mu : float
			mean task time
		priority : boolean
			prioritize the shortest tasks
	"""
	queue = Queue(env, num_servers, priority)
	i = 1
	yield env.timeout(-np.log(np.random.random()) / lamda)
	env.process(person(env,i,queue, mu, serv_t_dist))

	while i < max_persons:

		yield env.timeout(-np.log(np.random.random()) / lamda)

		i += 1
		env.process(person(env,i,queue, mu, serv_t_dist))

def simulation(servers, rho, priority, max_persons, serv_t_dist):
	global wt
	wt = []

	env = simpy.Environment()

	mu = 1
	lamda_eff = mu * rho * servers
	

	env.process(setup(env, servers,  lamda_eff, mu, priority, max_persons, serv_t_dist))

	env.run(None)
	return wt 

def compute_avg_wt(servers, rho_l, max_persons, sims, priority=False, serv_t_dist = "M", truncation = 0):
	"""
	compute the average waiting time for a scala of
	servers and values for lamda and plot the results

	Args:
		servers : list
			number of servers to conduct the simulation with
		lamda : list
			values for lamda to conduct the simulation with
		mu : float
			mean task time
		max_persons : int
			max persons in system
		t_lim : int
			max time simulated, stops all processes after
		sims : int
			number of simulation for each combination (servers, lamda)
	"""

	df = pd.DataFrame()


	for n_servers in servers:
		server_wt = []
		conf_int = []
		for rho in rho_l:
			print(n_servers, rho)
			mean_wt = []
			for _ in tqdm.tqdm(range(sims)):
				waiting_time = simulation(n_servers, rho, priority, max_persons, serv_t_dist)
				mean_wt.append(np.mean(waiting_time[truncation:]))
			new_row = {"servers":n_servers, "rho":rho, "mean_wt":mean_wt}
			df = df.append(new_row, ignore_index = True)
			server_wt.append(np.mean(mean_wt))
			conf_int.append(1.96 * np.std(mean_wt, ddof=1) / np.sqrt(sims))
		
		plt.errorbar(rho_l, server_wt, yerr = conf_int, label='{} servers, servicet dist = {}'.format(n_servers, serv_t_dist))
		plt.xlabel("rho [-]")
		plt.ylabel("mean waiting time [-]")
	plt.legend()
	return df

def wt_distribution(servers, rho_l, max_persons, sims_l, priority=False, serv_t_dist = "M"):
	"""
	compute the average waiting time for a scala of
	servers and values for lamda and plot the results

	Args:
		servers : int
			number of servers to conduct the simulation with
		rho_l : list
			values for rho to conduct the simulation with
		max_persons : int
			max persons in system
		max_persons : list
			maximum number of persons to arrive
		sims_l : int
			number of simulations for each combination 
	"""
	


	for rho in rho_l:
		server_wt = []
		for max_p in max_persons:
			print(max_p)
			for sims in sims_l:
				waiting_time_dist = []
				waiting_time_dist_trunc = []
				for _ in range(sims):
					waiting_time = simulation(servers, rho, priority, max_p, serv_t_dist)
					waiting_time_dist.append(np.mean(waiting_time))
					
					waiting_time_dist_trunc.append(np.mean(waiting_time[int(max_p * 0.3):]))

				# plt.hist(waiting_time_dist, label="max_p: {}".format(max_p), alpha = 0.3, density=True)
				plt.hist(waiting_time_dist, label="rho: {}".format(rho), alpha = 0.3, density=True)
				
				# plt.hist(waiting_time_dist_trunc, label="max_p: {}, truncated".format(max_p), alpha = 0.3, density=False)
				plt.xlabel("average waiting time [-]", fontsize = 16)
				plt.ylabel("density [-]", fontsize = 16)
	# 		server_wt.append(np.mean(mean_wt))
		
	# 	plt.plot(lamda, server_wt, label='{} servers'.format(n_servers))
	# 	plt.xlabel("lamda [-]")
	# 	plt.ylabel("mean waiting time [-]")
	plt.legend()

def convergence_avg_wt(servers, rho_l, sims, priority=False, serv_t_dist = "M"):
	"""
	compute the convergence of the waiting time

	Args:
		servers : list
			number of servers to conduct the simulation with
		lamda : list
			values for lamda to conduct the simulation with
		mu : float
			mean task time
		max_persons : int
			max persons in system
		t_lim : int
			max time simulated, stops all processes after
		sims : int
			number of simulation for each combination (servers, lamda)
	"""

	n_servers = servers[0]
	rho = rho_l[0]

	epsilon = 0.1
	n_repetitions = 10
	max_persons_converged = []
	last_wt_converge_value = []
	for i in tqdm.tqdm(range(n_repetitions)):
		max_persons = 100
		mean_wt = [2**32, 2**16]
		while abs(mean_wt[-1] - mean_wt[-2]) > epsilon:
			sim_mean_wt = []
			for _ in range(sims):
				waiting_time = simulation(n_servers, rho, priority, max_persons, serv_t_dist)

				sim_mean_wt.append(np.mean(wt))
			mean_wt.append(np.mean(sim_mean_wt))
			max_persons += 100

		# trim first two values from mean_wt
		mean_wt = mean_wt[2:]
		max_persons_converged.append(max_persons)
		last_wt_converge_value.append(mean_wt[-1])

	print(np.mean(max_persons_converged), np.mean(last_wt_converge_value))
	return max_persons_converged, last_wt_converge_value


if __name__ == "__main__":

	# lambda = arrival rate into the system as s whole
	# mu = the capacity of each of n equal servers
	# rho = system load. In a single system it will be rho = lambda/mu
	# in principle rho < 1 to prevent overload. For multiple servers n rho = lambda/(mu*n) < 1

	##################################################################################
	# average waiting time for n = [1, 2, 4] for lamda between 0.8 and 1.2
	##################################################################################
	global wt
	wt = []

	servers_l = [1, 2, 4]
	rho_l1 = np.linspace(0.1, 0.5, 5)
	rho_l2 = np.linspace(0.6, 0.95, 15)
	rho_l = np.concatenate((rho_l1, rho_l2), axis = 0)	
	max_persons = 20000
	sims = 100
	truncation = 6000

	# M/M/n queue without priority
	data = compute_avg_wt(servers_l, rho_l, max_persons, sims, truncation = truncation)
	data.to_csv("M124-p{}-t{}-sims{}_2.csv".format(max_persons, truncation, sims))


	# # M/M/1 queue with shortest job first scheduling
	data = compute_avg_wt([1], rho_l, max_persons, sims, priority=True, truncation = truncation)
	data.to_csv("M124-p{}-t{}-sims{}-prio_2.csv".format(max_persons, truncation, sims))

	# servers_l = [1]
	# rho_l = [0.9]

	# sims = 50
	# convergence_avg_wt(servers_l, rho_l, sims)



	# servers_l = [1, 2, 4]
	# rho_l = np.linspace(0.6, 1, 6)
	# max_persons = 200
	# sims = 50

	# M/M/n queue without priority
	# plt.figure()
	# compute_avg_wt(servers_l, rho_l, max_persons, sims, serv_t_dist = "M")

	# plt.figure()
	# data = compute_avg_wt(servers_l, rho_l, max_persons, sims, serv_t_dist = "D", truncation = truncation)
	# data.to_csv("D124-p{}-t{}-sims{}.csv".format(max_persons, truncation, sims))

	# plt.figure()
	# data = compute_avg_wt(servers_l, rho_l, max_persons, sims, serv_t_dist = "LT")
	# data.to_csv("LT124-p{}-t{}-sims{}.csv".format(max_persons, truncation, sims))


	##################################################################################
	# distribution of mean waiting times
	# used to determine whether the system has entered the steady state
	# is dependent on sims, rho and max_persons
	##################################################################################

	# servers = 4
	# rho_l = [0.5, 0.7, 0.95]
	# max_persons = [1000]
	# sims = [100]
	# wt_distribution(servers, rho_l, max_persons, sims, serv_t_dist = "M")
	# plt.xlim(0,6)
	# plt.legend(fontsize=16)

	plt.show()



