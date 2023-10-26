import React, { useRef, useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import { useTracker } from 'meteor/react-meteor-data';

import { Timeline as VisTimeline } from 'vis-timeline/peer';
import { DataSet as VisDataSet } from 'vis-data/peer';
import "vis-timeline/styles/vis-timeline-graph2d.css";

import { SchedulesCollection } from '/imports/api/schedules';

import moment from 'moment';

export const Schedule = () => {

  let { id } = useParams();
  const schedule = useTracker(() => SchedulesCollection.findOne(new Meteor.Collection.ObjectID(id)), [id])

  const items = useRef(new VisDataSet());
  const groups = useRef(new VisDataSet());

  const vistimeline = useRef(null);
  const container = useRef(null);

  const [showExternal, setShowExternal] = useState(false);

  // initialize visjs timeline widget
  useEffect(() => {
    let options = {
      min: moment().startOf('day'),
      showCurrentTime: false,
      showMajorLabels: false,
      timeAxis: {scale: 'second', step: 1},
      margin: { item: { horizontal: 0 } },
      groupOrder: "content", // groupOrder can be a property name or a sorting function
    };

    vistimeline.current = new VisTimeline(container.current);
    vistimeline.current.setOptions(options);
    vistimeline.current.setGroups(groups.current);
    vistimeline.current.setItems(items.current);

  }, [])

  // synchronize schedule from database to visjs DataSets
  useEffect(() => {
    if (!schedule) { return }

    items.current.clear()
    groups.current.clear()

    for (let key in schedule.y) {
      // match "(i, j)" tuples
      const re = /\(\s*(\d+)\s*\,\s*(\d+)\)/;

      const node = key.match(re)[1];
      const vehicle = key.match(re)[2];
      const start = moment().startOf('day').add(schedule.y[key], 's');

      let ptime = null;
      if (Object.hasOwn(schedule, 'ptime')) {
        // global fixed ptime for all jobs
        ptime = schedule.ptime;
      } else if (Object.hasOwn(schedule, 'ptimes')) {
        // each job has its own ptime
        ptime = schedule.ptimes[key];
      } else {
        throw "No processing times specified."
      }
      const end = start.clone().add(ptime, 's')

      // Create groups corresponding to machines (if not yet exists).
      let style = "";
      let skip = false;
      if ( Object.hasOwn(schedule, 'entrypoints') && Object.hasOwn(schedule, 'exitpoints') ) {
        if (schedule.entrypoints.includes(Number(node))) {
          style += "color: grey; background-color: #eeeeff;"
          skip = skip || !showExternal;
        }
        if (schedule.exitpoints.includes(Number(node))) {
          style += "color: grey; background-color: #ffeeee;"
          skip = skip || !showExternal;
        }
      }
      if (!skip) {
        groups.current.update([{ id: node, content: node, style: style }])
      }

      // Create items from y variables.
      items.current.add({
        id: key,
        group: node,
        content: vehicle,
        start: start,
        end: end,
        type: "range",
        style: style,
      })
    }

    vistimeline.current.fit();
  }, [schedule, showExternal]);

  return (
    <div>
      <h1>Schedule</h1>
      <label for="showExternal">Show External Nodes </label>
      <input id="showExternal" type="checkbox" checked={showExternal} onChange={e => setShowExternal(e.target.checked)} />
      <br />
      <br />
      <div ref={container}/>
    </div>
  )

};
