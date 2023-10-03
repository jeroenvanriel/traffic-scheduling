import { Mongo } from 'meteor/mongo';

export const SchedulesCollection = new Mongo.Collection('schedules', { 'idGeneration': 'MONGO' });
