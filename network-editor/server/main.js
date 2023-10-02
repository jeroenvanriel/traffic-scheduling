import { Meteor } from 'meteor/meteor';
import { NetworksCollection } from '/imports/api/networks';

async function insertNetwork() {
  await NetworksCollection.insertAsync({
    name: "example1",
    directed: true,
    multigraph: false,
    nodes: [
      {'color': 'red', 'vehicle': 1, 'id': 0},
      {'color': 'red', 'vehicle': 0, 'id': 1},
      {'color': 'red', 'vehicle': 0, 'id': 2},
      {'color': 'red', 'vehicle': 0, 'id': 3},
      {'color': 'red', 'vehicle': 0, 'id': 4},
      {'color': 'red', 'vehicle': 0, 'id': 5},
      {'color': 'red', 'vehicle': 0, 'id': 6},
      {'color': 'red', 'vehicle': 0, 'id': 7},
      {'vehicle': 2, 'id': 8},
      {'vehicle': 0, 'id': 9},
      {'vehicle': 0, 'id': 10},
      {'vehicle': 0, 'id': 11},
      {'vehicle': 0, 'id': 12},
      {'vehicle': 0, 'id': 13},
    ],
    edges: [
      {'active': true, 'from': 0, 'to': 1},
      {'active': true, 'from': 1, 'to': 2},
      {'active': true, 'from': 2, 'to': 3},
      {'active': true, 'from': 3, 'to': 4},
      {'active': true, 'from': 4, 'to': 5},
      {'active': true, 'from': 5, 'to': 6},
      {'active': true, 'from': 5, 'to': 12},
      {'active': true, 'from': 6, 'to': 7},
      {'active': true, 'from': 8, 'to': 9},
      {'active': true, 'from': 9, 'to': 10},
      {'active': true, 'from': 10, 'to': 11},
      {'active': false, 'from': 11, 'to': 5},
      {'active': true, 'from': 12, 'to': 13},
    ],
  });
}

Meteor.startup(async () => {
  if (await NetworksCollection.find().countAsync() === 0) {
    await insertNetwork();
  }

  // TODO: The following is currently not necessary because autopublish is
  // enabled To enable/disable this: `meteor add/remove autopublish`
  //
  // We publish the entire Networks collection to all clients.
  // In order to be fetched in real-time to the clients
  // Meteor.publish("networks", function () {
  //   return NetworksCollection.find();
  // });
});

Meteor.methods({
  'networks.update'({ name, nodes, edges }) {
    const net = NetworksCollection.findOne({ name: name });
    if (net) {
      // update
      NetworksCollection.update(net._id, {
        $set: { nodes: nodes, edges: edges }
      });
    } else {
      // create new
      const defaults = {
        directed: true,
        multigraph: false,
      }
      console.log('adding', edges)
      NetworksCollection.insert({
        ...defaults,
        name: name,
        nodes: nodes,
        edges: edges,
      });
    }
  }
})
