import React from 'react';
import { Link } from 'react-router-dom';

export const Home = () => {
  return (
    <div>
      <h2>Traffic Scheduling Tools</h2>
      <ul>
        <li> <Link to='/network'>Networks</Link> </li>
        <li> <Link to='/schedule'>Schedules</Link> </li>
      </ul>
    </div>
  )
}
