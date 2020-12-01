# Assignment 2
# Stochastic simulation DES simulation assignment
# Sjoerd Terpstra
# Coen Lenting

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
		Args:
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
		Args:
			task_time : float
				time needed to perform the task
		"""
		yield self.env.timeout(task_time)


def person(env, name, queue, mu, serv_t_dist):
	"""
	a function to represent a person joining the queue and
	performing a task when specified. Appends the waiting time to the global list wt.

	Args:
		name : int
			unique representation of a person
		queue : Queue
			the queue joined
		mu : float
			mean task time
		serv_t_dist : str
			service time distribution
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

	# handle priority
	if not queue.priority:
		with queue.server.request() as request:
			# request a position at the queue
			yield request

			# request accepted, server entered
			enter = env.now
			yield env.process(queue.task(task_time))

			# waiting time
			wt.append(enter - arrival)
	else:
		with queue.server.request(priority = task_time, preempt=True) as request:
			yield request

			enter = env.now
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
		max_persons : int
			max arrivals handled
		serv_t_dist : str
			service time distribution
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
	"""
	performs a simulation with the specified parameters

	Args:
		servers : int
			number of servers
		rho : float
			system load
		priority : boolean
			prioritize the shortest tasks
		max_persons : int
			max arrivals handled
		serv_t_dist : str
			service time distribution
	"""
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
		rho_l : list
			values for rho to conduct the simulation with
		max_persons : int
			max arrivals handled
		sims : int
			number of simulation for each combination (servers, lamda)
		priority : boolean
			prioritize the shortest tasks
		serv_t_dist : str
			service time distribution
		truncation : int
			number of waiting times truncated from the start
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
		max_persons : list
			maximum number of persons to arrive
		sims_l : int
			number of simulations for each combination
		priority : boolean
			prioritize the shortest tasks
		serv_t_dist : str
			service theime distribution
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

				if len(rho_l) > 1:
					plt.hist(waiting_time_dist, label="rho: {}".format(rho), alpha = 0.3, density=True)
				else:
					plt.hist(waiting_time_dist, label="max_p: {}".format(max_p), alpha = 0.3, density=True)
				plt.xlabel("average waiting time [-]", fontsize = 16)
				plt.ylabel("density [-]", fontsize = 16)

	plt.legend()

if __name__ == "__main__":

	global wt
	wt = []

	# parameters
	servers_l = [1, 2, 4]
	rho_l1 = np.linspace(0.1, 0.5, 2)
	rho_l2 = np.linspace(0.6, 0.95, 3)
	rho_l = np.concatenate((rho_l1, rho_l2), axis = 0)
	max_persons = 1000
	sims = 30
	truncation = 300

	plt.figure()

	# M/M/n queue without priority
	data = compute_avg_wt(servers_l, rho_l, max_persons, sims, truncation = truncation)
	data.to_csv("M124-p{}-t{}-sims{}_2.csv".format(max_persons, truncation, sims))

	# M/M/1 queue with shortest job first scheduling
	data = compute_avg_wt([1], rho_l, max_persons, sims, priority=True, truncation = truncation)
	data.to_csv("M124-p{}-t{}-sims{}-prio_2.csv".format(max_persons, truncation, sims))
	plt.legend(fontsize=16)

	# M/D/n queue
	plt.figure()
	data = compute_avg_wt(servers_l, rho_l, max_persons, sims, serv_t_dist = "D", truncation = truncation)
	data.to_csv("D124-p{}-t{}-sims{}_2.csv".format(max_persons, truncation, sims))
	plt.legend(fontsize=16)

	# M/LT/n queue
	plt.figure()
	data = compute_avg_wt(servers_l, rho_l, max_persons, sims, serv_t_dist = "LT")
	data.to_csv("LT124-p{}-t{}-sims{}_2.csv".format(max_persons, truncation, sims))
	plt.legend(fontsize=16)

	##################################################################################
	# distribution of mean waiting times
	# used to determine whether the system has entered the steady state
	# is dependent on sims, rho and max_persons
	##################################################################################

	plt.figure()
	rho_l = [0.95]
	max_persons = [100, 1000, 2000]
	sims = [100]
	servers = 4
	wt_distribution(servers, rho_l, max_persons, sims, serv_t_dist = "M")
	plt.legend(fontsize=16)

	plt.figure()
	rho_l = [0.5, 0.7, 0.95]
	max_persons = [100]
	wt_distribution(servers, rho_l, max_persons, sims, serv_t_dist = "M")
	plt.legend(fontsize=16)
	plt.show()
