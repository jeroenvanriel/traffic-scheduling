import React from 'react';
import { createRoot } from 'react-dom/client';
import { Meteor } from 'meteor/meteor';
import { Network } from '/imports/ui/Network.jsx'
import { NetworkList } from '/imports/ui/NetworkList';

import {
  createBrowserRouter,
  RouterProvider
} from 'react-router-dom';

const router = createBrowserRouter([
  {
    path: "/",
    element: <NetworkList />,
  },
  {
    path: "/network/:name",
    element: <Network />,
  }
])

Meteor.startup(() => {
  const container = document.getElementById('react-target');
  const root = createRoot(container);
  root.render(<RouterProvider router={router} />);
});
