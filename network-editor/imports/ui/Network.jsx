import "./network.css";
import React, { useState, useRef, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import { useTracker } from 'meteor/react-meteor-data';
import { Meteor } from 'meteor/meteor';

import { NetworksCollection } from '/imports/api/networks';

import { Network as VisNetwork } from 'vis-network/peer';
import { DataSet as VisDataSet } from 'vis-data/peer';


// width/height of the grid for adding nodes
const gridSize = 100;


export const Network = () => {
  let { name } = useParams();
  const network = useTracker(() => NetworksCollection.findOne({ name: name }), [name]);

  const [nextId, setNextId] = useState(1);
  const [selected, setSelected] = useState(null);

  const visnetwork = useRef(null);
  const container = useRef(null);

  const nodes = useRef(new VisDataSet());
  const edges = useRef(new VisDataSet());

  const [chainAdding, setChainAdding] = useState(false);

  // initialize visjs network widget
  useEffect(() => {
    const data = {
      nodes: nodes.current,
      edges: edges.current
    };

    nodes.current.on("*", function (event, properties, senderId) {
      // prevent infinite loop by checking whether this update came from here
      if (senderId == 'labelcomputation') { return }
      // re-compute labels
      nodes.current.updateOnly(nodes.current.map(
        (node) => ({ ...node, label: getLabel(node) })
      ), 'labelcomputation');
    })

    const options = {
      height: '600px',
      manipulation: {
        enabled: true,
        initiallyActive: true,
        addNode: false,
        addEdge: true,
        editEdge: true,
        deleteNode: true,
        deleteEdge: true,
      },
      physics: {
        enabled: false,
      },
      edges: {
        arrows: {
          to: { enabled: true }
        },
        smooth: {
          enabled: false,
        }
      },
    }

    visnetwork.current = new VisNetwork(container.current, data, options);

    visnetwork.current.on("selectNode", function (params) {
      // Not sure whether it's possible to select multiple nodes, so we just
      // pick the first one as the 'last selected'
      setSelected(params.nodes[0]);
    });
  }, [])

  // add click handler for adding nodes
  useEffect(() => {
    visnetwork.current.off("click")
    visnetwork.current.on("click", function (params) {
      // check if current selection is empty
      if (params.nodes.length + params.edges.length == 0) {
        addNode(nextId, params.pointer.canvas.x, params.pointer.canvas.y,
                chainAdding ? selected : null);

        setSelected(nextId);
        setNextId(id => id + 1);
      }
    });
  }, [chainAdding, selected])

  // update delete node handler to correct selection
  // TODO: maybe better to support selected==null as state
  useEffect(() => {
    visnetwork.current.setOptions({
      manipulation: {
        deleteNode: (selection, callback) => {
          callback(selection);

          // when currently selected node was removed, update selection to node
          // with max id
          if (selection.nodes.includes(selected)) {
            const maxId = nodes.current.get().reduce((max, cur) => Math.max(max, cur.id), -Infinity);
            setSelected(maxId);
          }
        }
      }
    });
  }, [selected]);

  // synchronize network from database to visjs DataSets
  useEffect(() => {
    if (network) {
      nodes.current.clear()
      nodes.current.update(network.nodes);
      edges.current.clear()
      edges.current.update(network.edges);

      // make sure new id's are unique
      const maxId = network.nodes.reduce((max, cur) => Math.max(max, cur.id), -Infinity);
      setSelected(maxId);
      setNextId(maxId + 1);
    }
  }, [network]);

  function getLabel(node) {
    return `id: ${node.id}\nveh: ${node.vehicle}`;
  }

  function addNode(id, x, y, from) {
    // add a new node
    nodes.current.update({ id: id, label: id.toString(), vehicle: 0 });

    if (from) {
      edges.current.update({ from: from, to: id })
    }

    // snap new node to grid
    const [grid_x, grid_y] = getGridPosition(x, y)
    visnetwork.current.moveNode(id, grid_x, grid_y);
  }

  function snapToGrid() {
    nodes.current.map(node => {
      let {x, y} = visnetwork.current.getPosition(node.id);
      [x, y] = getGridPosition(x, y);
      visnetwork.current.moveNode(node.id, x, y);
    });
  }

  function getGridPosition(x, y) {
    return [
      x = Math.round(x / gridSize) * gridSize,
      y = Math.round(y / gridSize) * gridSize,
    ];
  }

  function save() {
    // fix the positions
    visnetwork.current.storePositions();

    // send to server for saving to mongodb
    Meteor.call('networks.update', {
      name: name,
      nodes: nodes.current.get({ fields: ['id', 'label', 'x', 'y', 'vehicle', 'color'] }),
      edges: edges.current.get({ fields: ['from', 'to', 'active'] }),
    }, (err, _) => {
      if (err) { alert(err); }
      else { console.log("successfully saved") }
    });
  }

  return (
    <div>
      <h1>{network && network.name}</h1>
      <button onClick={snapToGrid}>snap to grid</button>
      <button onClick={save}>save</button>
      { chainAdding ?
        <button onClick={() => setChainAdding(false)}>end chain adding</button>
      : <button onClick={() => setChainAdding(true)}>start chain adding</button> }
      <div ref={container}/>
      {/* pass key to reset when selection changes */}
      { nodes.current && selected && <NodeEditor nodes={nodes.current} selected={selected} key={selected} /> }
    </div>
  );
};


const NodeEditor = ({ nodes, selected }) => {
  const node = nodes.get(selected);

  const [vehicle, setVehicle] = useState(node.vehicle);
  const [color, setColor] = useState(node.color ? node.color : '');

  const handleSubmit = e => {
    e.preventDefault();

    node.vehicle = vehicle;
    if (color != '') {
      node.color = color;
    } else {
      // reset to default visjs color by removing the color attribute
      node.color = null;
    }

    nodes.update(node);
  }

  return (
    <form onSubmit={handleSubmit}>
      <h2>Edit Node {node.label}</h2>

      <label htmlFor="vehicle">Vehicle:</label>
      <input id="vehicle" type="text"
             value={vehicle}
             onChange={(e) => setVehicle(e.target.value)}
      />

      <label htmlFor="color">Color:</label>
      <input id="color" type="text"
             value={color}
             onChange={(e) => setColor(e.target.value)}
      />

      <button type="submit">Apply</button>
    </form>
  )
}
