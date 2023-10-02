import React, { useRef, useEffect } from 'react';

import { Timeline as VisTimeline } from 'vis-timeline/peer';
import { DataSet as VisDataSet } from 'vis-data/peer';
import "vis-timeline/styles/vis-timeline-graph2d.css";

import moment from 'moment';

export const Timeline = () => {

  const vistimeline = useRef(null);
  const container = useRef(null);

  // initialize visjs timeline widget
  useEffect(() => {
    let now = moment().minutes(0).seconds(0).milliseconds(0);
    let groupCount = 3;
    let itemCount = 20;

    // create a data set with groups
    let names = ["John", "Alston", "Lee", "Grant"];
    let groups = new VisDataSet();
    for (let g = 0; g < groupCount; g++) {
        groups.add({ id: g, content: names[g] });
    }

    // create a dataset with items
    let items = new VisDataSet();
    for (let i = 0; i < itemCount; i++) {
        let start = now.clone().add(Math.random() * 20, "hours");
        let end = start.clone().add(4, "hours");
        let group = Math.floor(Math.random() * groupCount);
        items.add({
            id: i,
            group: group,
            content:
            "item " +
            i +
            ' <span style="color:#97B0F8;">(' +
            names[group] +
            ")</span>",
            start: start,
            end: end,
            type: "range",
        });
    }

    // create visualization
    let options = {
        groupOrder: "content", // groupOrder can be a property name or a sorting function
    };

    vistimeline.current = new VisTimeline(container.current);
    vistimeline.current.setOptions(options);
    vistimeline.current.setGroups(groups);
    vistimeline.current.setItems(items);

  }, [])

  return (
    <div>
      <h1>Timeline</h1>
      <div ref={container}/>
    </div>
  )

};
