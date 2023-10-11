import networkx as nx
import numpy as np
import math, random
import argparse

# read command line arguments
parser = argparse.ArgumentParser(description='Generate routes and vehicles.')
parser.add_argument('network', help='network name as found in mongo')
parser.add_argument('-o', dest='output', help='output file name')
parser.add_argument('-p', type=int, dest='processing_time', default=1, help='processing time')
parser.add_argument('-s', type=int, dest='switch_over_time', default=1, help='switch-over time')
parser.add_argument('-n', type=int, dest='total_vehicles', default=1, help='total number of vehicles')
args = parser.parse_args()

# establish mongodb database connection
from pymongo import MongoClient
client = MongoClient("mongodb://127.0.0.1:3001/meteor")
db = client.meteor

def distance(p, q):
    # scale for converting pixel-distance (vis-network) to distance-units used in the MIP model
    SCALE = 100
    return math.sqrt((p['x'] - q['x'])**2 + (p['y'] - q['y'])**2) / SCALE


def load_network(name):
    network = db.networks.find_one({'name': name})

    if not network:
        raise Exception('network not found in database')

    G = nx.DiGraph()

    for node in network['nodes']:
        G.add_node(node['id'], x=node['x'], y=node['y'])

    for edge in network['edges']:
        p = G.nodes[edge['from']]
        q = G.nodes[edge['to']]
        G.add_edge(edge['from'], edge['to'], distance=distance(p, q))

    return G


network = load_network(args.network)
adj = nx.adjacency_matrix(network, weight='distance')

# identify external nodes
entry_nodes = []
exit_nodes = []
for node in network.nodes:
    if network.in_degree(node) == 0 and network.out_degree(node) == 1:
        entry_nodes.append(node)
    if network.in_degree(node) == 1 and network.out_degree(node) == 0:
        exit_nodes.append(node)


# construct a route for every entrypoint
routes = []
for begin in entry_nodes:
    # construct route from begin by picking random neighbors until some exit is reached
    route = [begin]

    while True:
        next_node = random.choice(list(network.adj[route[-1]]))
        route.append(next_node)
        if next_node in exit_nodes:
            break

    routes.append(route)

print("generated routes:")
print('\n'.join(map(str, routes)))


# add some vehicles
# TODO: add randomness and parameterize
# currently: adding vehicles 'round-robin'-like w.r.t. routes
vehicles = []
route_release_dates = [0 for _ in routes]
current_route = 0

for _ in range(args.total_vehicles):
    vehicles.append({
        'route_id': current_route,
        'release_date': route_release_dates[current_route]
    })

    # random platoon split
    if random.random() < 0.1:
        route_release_dates[current_route] += 4
    else:
        route_release_dates[current_route] += 1

    current_route = (current_route + 1) % len(routes)

# sort vehicles by route
vehicles.sort(key=lambda veh: veh['route_id'])


# write to file
if args.output is None:
    args.output = args.network + '.txt'
with open(args.output, 'w') as f:
    f.write(f'# generated for network: {args.network}\n\n')

    f.write('# number of vehicles, number of nodes\n')
    f.write(str(len(vehicles)) + " " + str(network.number_of_nodes()) + "\n")

    f.write('# processing time, switch-over time\n')
    f.write(str(args.processing_time) + " " + str(args.switch_over_time) + "\n")

    f.write('\n# adjacency matrix\n')
    np.savetxt(f, adj.todense(), fmt='%1d')

    f.write('\n# list of vehicles\n')
    # each line contains:
    # - release date
    # - route as list of nodes
    for vehicle in vehicles:
        line = str(vehicle['release_date']) + '  '
        line += ' '.join(map(str, routes[vehicle['route_id']]))
        line += "\n"

        f.write(line)
