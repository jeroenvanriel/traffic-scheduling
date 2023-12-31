import React from 'react';
import { createRoot } from 'react-dom/client';
import { Meteor } from 'meteor/meteor';
import { Home } from '/imports/ui/Home.jsx';
import { Network } from '/imports/ui/Network.jsx'
import { NetworkList } from '/imports/ui/NetworkList';
import { Schedule } from '/imports/ui/Schedule';
import { ScheduleList } from '/imports/ui/ScheduleList';

import {
  createBrowserRouter,
  RouterProvider
} from 'react-router-dom';

const router = createBrowserRouter([
  {
    path: "/",
    element: <Home />,
  },
  {
    path: "/network",
    element: <NetworkList />,
  },
  {
    path: "/network/:name",
    element: <Network />,
  },
  {
    path: "/schedule",
    element: <ScheduleList />,
  },
  {
    path: "/schedule/:id",
    element: <Schedule />,
  },
])

Meteor.startup(() => {
  const container = document.getElementById('react-target');
  const root = createRoot(container);
  root.render(<RouterProvider router={router} />);
});
