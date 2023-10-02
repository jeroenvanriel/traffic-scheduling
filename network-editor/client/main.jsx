import React from 'react';
import { createRoot } from 'react-dom/client';
import { Meteor } from 'meteor/meteor';
import { Network } from '/imports/ui/Network.jsx'
import { NetworkList } from '/imports/ui/NetworkList';

import {
  createBrowserRouter,
  RouterProvider
} from 'react-router-dom';
import { Timeline } from '../imports/ui/Schedule';

const router = createBrowserRouter([
  {
    path: "/",
    element: <NetworkList />,
  },
  {
    path: "/network/:name",
    element: <Network />,
  },
  {
    path: "/timeline",
    element: <Timeline />,
  }
])

Meteor.startup(() => {
  const container = document.getElementById('react-target');
  const root = createRoot(container);
  root.render(<RouterProvider router={router} />);
});
