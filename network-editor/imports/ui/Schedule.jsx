import React, { useRef, useEffect } from 'react';
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

  // synchronize network from database to visjs DataSets
  useEffect(() => {
    if (!schedule) { return }

    items.current.clear()
    groups.current.clear()

    const ptime = schedule.ptime;

    for (let key in schedule.y) {
      // match "(i, j)" tuples
      const re = /\(\s*(\d+)\s*\,\s*(\d+)\)/;

      const machine = key.match(re)[1];
      const job = key.match(re)[2];
      const start = moment().startOf('day').add(schedule.y[key], 's');
      const end = start.clone().add(ptime, 's')

      // Create groups corresponding to machines (if not yet exists).
      groups.current.update([{ id: machine, content: machine }])

      // Create items from y variables.
      items.current.add({
        id: key,
        group: machine,
        content: job,
        start: start,
        end: end,
        type: "range",
      })
    }

    vistimeline.current.fit();
  }, [schedule]);

  return (
    <div>
      <h1>Timeline</h1>
      <div ref={container}/>
    </div>
  )

};
