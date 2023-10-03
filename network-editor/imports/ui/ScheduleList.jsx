import React from 'react';
import { useTracker } from 'meteor/react-meteor-data';
import { Link } from 'react-router-dom';
import { SchedulesCollection } from '/imports/api/schedules';

import moment from 'moment';

const ScheduleName = ({schedule}) => {
  return <li key={schedule._id}>
           <Link to={`/schedule/${schedule._id._str}`}>
             {moment(schedule.date).format('DD-MM-YYYY hh:mm')}
            </Link>
         </li>
}

export const ScheduleList = () => {
  const schedules = useTracker(() => SchedulesCollection.find().fetch());

  console.log(schedules)

  return (
    <div>
      <h2>Schedules</h2>
      <ul>
        { schedules.map(sched => <ScheduleName key={sched._id._str} schedule={sched}/>) }
      </ul>
    </div>
  )
}
