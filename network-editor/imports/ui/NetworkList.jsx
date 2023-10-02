import React from 'react';
import { useTracker } from 'meteor/react-meteor-data';
import { Link } from 'react-router-dom';
import { NetworksCollection } from '/imports/api/networks';

const NetworkName = ({network}) => {
  return <li key={network._id}><Link to={`/network/${network.name}`}>{network.name}</Link></li>
}

export const NetworkList = () => {
  const networks = useTracker(() => NetworksCollection.find({}).fetch());

  return (
    <div>
      <h2>Networks</h2>
      <ul>
        { networks.map(net => <NetworkName key={net._id} network={net}/>) }
      </ul>
    </div>
  )
}
