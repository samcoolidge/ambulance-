import json
import random
from hps.clients import SocketClient
import sys
import time
import numpy as np
import math
import copy
from sklearn.cluster import KMeans

HOST = '127.0.0.1'
PORT = 5000

class Player(object):
    def __init__(self, name):
        self.name = name
        self.client = SocketClient(HOST, PORT)
        self.client.send_data(json.dumps({'name': self.name}))


    def play_game(self):
        buffer_size_message = json.loads(self.client.receive_data(size=2048))
        buffer_size = int(buffer_size_message['buffer_size'])
        game_state = json.loads(self.client.receive_data(size=buffer_size))
        self.patients = {int(key): value for key, value in game_state['patients'].items()}
        self.hospitals = {int(key): value for key, value in game_state['hospitals'].items()}
        self.ambulances = {int(key): value for key, value in game_state['ambulances'].items()}

        # Get hospital locations and ambulance routes
        (hos_locations, amb_routes) = self.your_algorithm()
        response = {'hospital_loc': hos_locations, 'ambulance_moves': amb_routes}
        print('sending data')
        min_buffer_size = sys.getsizeof(json.dumps(response))
        print(min_buffer_size)
        print(response)

        buff_size_needed = 1 << (min_buffer_size - 1).bit_length()
        buff_size_needed = max(buff_size_needed, 2048)
        buff_size_message = {'buffer_size': buff_size_needed}
        self.client.send_data(json.dumps(buff_size_message))
        time.sleep(2)
        self.client.send_data(json.dumps(response))

        # Get results of game
        game_result = json.loads(self.client.receive_data(size=8192))
        if game_result['game_completed']:
            print(game_result['message'])
            print('Patients that lived:')
            print(game_result['patients_saved'])
            print('---------------')
            print('Number of patients saved = ' + str(game_result['number_saved']))
        else:
            print('Game failed run/validate ; reason:')
            print(game_result['message'])


    def your_algorithm(self):

        def get_pos_list():
            pos_list = []
            for p in self.patients:
                pos_list.append([self.patients[p]['xloc'],self.patients[p]['yloc']])
            return pos_list

        def parse_clusters(pos_list,labels):
            clusters = [[],[],[],[],[]]
            for p in range(len(pos_list)):
                index = labels[p]
                clusters[index].append(pos_list[p]+[self.patients[p]['rescuetime']]+[p])
            return clusters

        def get_total_urgency(clusters,centroids):
            urgency = []
            for c in range(5):
                centroid = centroids[c]
                cluster = clusters[c]
                distance = 0
                survival = 0
                for p in cluster:
                    distance += abs(p[0] - centroid[0]) + abs(p[1] - centroid[1])
                    #survival += p[2]
                    survival += p[2]/len(cluster)
                urgency.append(distance/float(survival))
            return urgency
        def distance(pos1,pos2):
            return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

        def get_next_patient(cluster,amb_pos,hosp_pos,riders,t,dis_bool,death_bool):
            if len(riders) == 4:
                return 0
            possibles = []
            distance_list = []
            death_list = []
            for patient in cluster:
                pos = [patient[0],patient[1]]
                d = distance(amb_pos,pos)
                time_to_save = d + distance(pos,hosp_pos) + 2
                rescue_times = [r[2] for r in riders] + [patient[2]]
                possible = True
                for rt in rescue_times:
                    if rt < time_to_save + t:
                        possible = False
                if possible:
                    possibles.append(patient)
                    distance_list.append(d)
                    death_list.append(patient[2] - t)

            if len(possibles) == 0:
                return 0
            if min(distance_list) < 3:
                return possibles[np.argmin(distance_list)]

            if dist_bool and not death_bool:
                return possibles[np.argmin(distance_list)]

            if death_bool and not dist_bool:
                return possibles[np.argmin(death_list)]

            if death_bool and dist_bool:
                index = np.random.choice([0,1],1,[.6,.4])[0]
                return[possibles[np.argmin(distance_list)],possibles[np.argmin(death_list)]][index]
            if len(possibles) > 30:
                dist_index = np.argsort(distance_list)[-15:]
                death_index = np.argsort(death_list)[-15:]
                index_list = list(dist_index) + list(death_index)
                possibles = [possibles[i] for i in index_list]
            #MESS AROUND WITH PROB_LIST
            prob_list = np.array([1/float(((distance_list[i])*death_list[i])) for i in range(len(possibles))])
            prob_list = list(prob_list/float(np.sum(prob_list)))
            index_list = list(range(len(possibles)))
            index = np.random.choice(index_list,1,prob_list)[0]
            #index = random.choice(index_list)
            return possibles[index]

        def converge(my_list,thresh):
            bool = True
            for l in my_list:
                for other in my_list:
                    if abs(l-other) > thresh:
                        bool = False
            return bool

        def get_u2a_ratio(cluster_urgency,hospitals_sorted,urgency,amb_num_list):

            u2a_ratio = []
            for i in range(5):
                index = list(cluster_urgency).index(i)
                hosp = hospitals_sorted[index]
                u2a_ratio.append(urgency[i]/float(amb_num_list[hosp]))
            return u2a_ratio


        def adjust_clusters(clusters,centroids,urgency,cluster_urgency,amb_num_list,hospitals_sorted):

            u2a_ratio = get_u2a_ratio(cluster_urgency,hospitals_sorted,urgency,amb_num_list)
            clusters_new = copy.deepcopy(clusters)

            bool = False
            t0 = time.time()
            while not bool:
                mean_u = np.mean(u2a_ratio)
                index_list = [i for i in range(5) if u2a_ratio[i] < mean_u]
                for c in index_list:
                    clust = clusters_new[c]
                    others = copy.deepcopy(clusters_new)
                    min_dist = 2000
                    for o in range(len(others)):
                        if o == c or o in index_list:
                            continue
                        for p in range(len(others[o])):
                            dist = distance(others[o][p],centroids[c])
                            if dist < min_dist:
                                min_index = [o,p]
                                min_dist = dist
                    clusters_new[c].append(copy.deepcopy(clusters_new[min_index[0]][min_index[1]]))
                    del clusters_new[min_index[0]][min_index[1]]

                urgency = get_total_urgency(clusters_new,centroids)
                cluster_urgency = np.argsort(urgency)
                u2a_ratio = get_u2a_ratio(cluster_urgency,hospitals_sorted,urgency,amb_num_list)
                bool = converge(u2a_ratio,.1)
                if time.time() - t0 > 1:
                    break

            return clusters_new

        res_hos = {}
        res_amb = {}

        res_hos_max_outer = {}
        res_amb_max_outer = {}
        off_count = 0
        saved_max_outer = 0
        previous_centroids = []
        for trials in range(40):
            res_hos_temp = {}
            pos_list = get_pos_list()
            kmeans = KMeans(init='k-means++', n_clusters=5, n_init=1)
            kmeans.fit(pos_list)
            centroids = np.empty(shape = (5,2))
            centroids_d = kmeans.cluster_centers_
            for i in range(len(centroids_d)):
                for j in range(len(centroids_d[i])):
                    centroids[i,j] = int(round(centroids_d[i][j]))
            clusters = parse_clusters(pos_list,kmeans.labels_)

            urgency = get_total_urgency(clusters,centroids)
            cluster_urgency = np.argsort(urgency)
            amb_num_list = []
            for i in range(5):
                amb_num_list.append(len(self.hospitals[i]['ambulances_at_start']))
            hospitals_sorted = np.argsort(amb_num_list)
            for i in range(5):
                centroid_x = int(centroids[cluster_urgency[i],0])
                centroid_y = int(centroids[cluster_urgency[i],1])
                int(centroids[cluster_urgency[i],1])
                res_hos_temp[int(hospitals_sorted[i])] = {'xloc':centroid_x,'yloc':centroid_y}

            max_rescue = 0
            for p in self.patients:
                if self.patients[p]['rescuetime'] > max_rescue:
                    max_rescue = self.patients[p]['rescuetime']


            print(cluster_urgency)
            print(hospitals_sorted)
            print()

            #u2a_ratio = get_u2a_ratio(cluster_urgency,hospitals_sorted,urgency,amb_num_list)

            for i in range(5):
                '''
                print("hospital location :" ,res_hos_temp[i])
                print("hospital amb number: " , amb_num_list[i])
                index = list(hospitals_sorted).index(i)
                clust_num = cluster_urgency[index]
                print("cluster number: " , clust_num)
                print("centroid:" , centroids[clust_num])
                print("urgency at loction: " , urgency[clust_num])
                #print("u2a: " , u2a_ratio[clust_num])
                print()
                '''

            cluster_urgency_copy = copy.deepcopy(cluster_urgency)
            print(len(clusters[0]))
            centroids_new = copy.deepcopy(centroids)
            urgency_new = copy.deepcopy(urgency)
            cluster_urgency_new = copy.deepcopy(cluster_urgency)
            amb_num_list_new = copy.deepcopy(amb_num_list)
            hospitals_sorted_new = copy.deepcopy(hospitals_sorted)

            clusters_new = copy.deepcopy(clusters)

            if trials < 20:
                clusters_new = adjust_clusters(clusters_new,centroids_new\
                ,urgency_new,cluster_urgency_new,amb_num_list_new,hospitals_sorted_new)

            print(len(clusters_new[0]))

            '''
            for i in range(5):
                print("hospital location :" ,res_hos_temp[i])
                print("hospital amb number: " , amb_num_list[i])
                index = list(hospitals_sorted).index(i)
                clust_num = cluster_urgency[index]
                print("cluster number: " , clust_num)
                print("centroid:" , centroids[clust_num])
                print("urgency at loction: " , urgency[clust_num])
                #print("u2a: " , u2a_ratio[clust_num])
                print()
            '''

            res_amb_max = {}
            saved_max=0
            t0 = time.time()
            t1 = 0
            max_time = 2
            count = 0

            while t1 < max_time:
                if count < 5 :
                    dist_bool = True
                    death_bool = False
                elif count < 10:
                    dist_bool = False
                    death_bool = True
                elif t1 < 2*max_time/float(4):
                    dist_bool = False
                    death_bool = False
                else:
                    dist_bool = True
                    death_bool = True
                count += 1
                res_amb_temp = {}
                saved = 0
                saved_riders = []
                for i in range(5):
                    clusters_copy = copy.deepcopy(clusters_new)
                    hosp = self.hospitals[hospitals_sorted[i]]
                    clust = clusters_copy[cluster_urgency_new[i]]
                    hosp_pos = centroids[cluster_urgency_new[i]]
                    for amb in hosp['ambulances_at_start']:
                        t = 0
                        route = []
                        riders = []
                        amb_pos = hosp_pos
                        while t<max_rescue:
                            while True:
                                next_patient = get_next_patient(clust,amb_pos,hosp_pos,riders,t,dist_bool,death_bool)
                                if next_patient == 0:
                                    break
                                riders.append(next_patient)
                                route.append('p'+str(next_patient[-1]))
                                clust.remove(next_patient)
                                t += distance(amb_pos,next_patient[0:2]) + 1
                                amb_pos = next_patient[0:2]
                            if len(riders) == 0:
                                break
                            route.append('h'+str(hospitals_sorted[i]))
                            t += distance(amb_pos,hosp_pos) + len(riders)
                            amb_pos = hosp_pos
                            saved += len(riders)
                            saved_riders += riders
                            riders = []

                        if route == []:
                            route = ['h0']
                        res_amb_temp[amb] = route


                #print("saved: ", saved)
                saved_riders = [r[-1] for r in saved_riders]
                #print(saved_riders)
                #print("number saved: " , len(saved_riders))

                if saved > saved_max:
                    saved_max = saved
                    res_amb_max = res_amb_temp
                t1 = time.time() - t0

            print("max saved with this cluster: " , saved_max)
            print("previous max: " ,saved_max_outer )
            if saved_max > saved_max_outer:
                saved_max_outer = saved_max
                res_amb_max_outer = res_amb_max
                res_hos_max_outer = res_hos_temp

        res_amb = res_amb_max_outer
        res_hos = res_hos_max_outer

        return (res_hos, res_amb)
