import simpy
import numpy as np 
import matplotlib.pyplot as plt 

# 1) Derive that the average waiting time is shorter for an M/M/2 queue 
#    with a system load rho and processor capacity mu than for a single
#    M/M/1 queue with the same load characteristics (n-fold lower arrival
#    rate)



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


def person(env, name, queue, mu):
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
	task_time = -np.log(np.random.random()) / mu
	print("{} arrives at queue {} with task time {}".format(name, arrival, task_time))

	# handle priority
	if not queue.priority:
		with queue.server.request() as request:
			# request a position at the queue
			yield request

			# request accepted, server entered
			enter = env.now
			print("{} is served at {}".format(name, enter))
			yield env.process(queue.task(task_time))

			# waiting time
			wt.append(enter - arrival)
	else:
		with queue.server.request(priority = task_time, preempt=True) as request:
			yield request

			enter = env.now
			print("{} is served at {}".format(name, enter))
			yield env.process(queue.task(task_time))

			wt.append(enter - arrival)

def setup(env, num_servers, lamda, mu, priority):
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
	i = 0
	while True:
		# compute time of arrival of next person
		yield env.timeout(-np.log(np.random.random()) / lamda)
		i += 1
		env.process(person(env,i,queue, mu))


def compute_avg_wt(servers, lamda, mu, t_lim, sims, priority = False):
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
		t_lim : int 
			max time simulated, stops all processes after
		sims : int
			number of simulation for each combination (servers, lamda)
	"""
	for n_servers in servers:
		server_wt = []
		for l in lamda:
			mean_wt = []
			for _ in range(sims):
				global wt
				wt = []

				env = simpy.Environment()
				env.process(setup(env, n_servers, l * n_servers, mu, priority))
				env.run(until=t_lim)
				mean_wt.append(np.mean(wt))

			server_wt.append(np.mean(mean_wt))

		plt.plot(lamda, server_wt, label='{} servers'.format(n_servers))
		plt.xlabel("lamda [-]")
		plt.ylabel("mean waiting time [-]")
	plt.legend()
	# plt.show()



if __name__ == "__main__":
	# M/M/c - FIFO
	# for c = [1, 2, 4]

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
	lamda_l = np.linspace(0.8, 1.2, 6)
	
	compute_avg_wt(servers_l, lamda_l, 1, 40, 100)

	# M/M/1 queue with shortest job first scheduling
	compute_avg_wt([1], lamda_l, 1, 40, 100, True)
	plt.show()
	
	# global wt
	# wt = []
	# env = simpy.Environment()
	# env.process(setup(env, 1,  1, 1, True))
	# env.run(until=20)
	# print(env.active_process)





