# %%
from spidb import spidb, normalization, visualizer
import pandas as pd
import numpy as np
from datetime import timedelta
from scipy.signal import find_peaks
import multiprocessing as mp


def generate_metrics(sample):

    db = spidb.Database(r"data/spi.db")

    # sensor = db.session.get(spidb.Sensor, 1)

    sample = db.session.get(spidb.Sample, sample)

    # sample = db.session.query(spidb.Sample).filter(spidb.Sample.id == s).first()
    a = db.get_audio(
        start=sample.datetime,
        end=sample.datetime + timedelta(seconds=60),
        sensor=sample.sensor,
        channel_number=sample.channel.number,
    )
    a.fade_in(1, overwrite=True)
    a.fade_out(1, overwrite=True)

    nspa = normalization.calculate_nspa(a, filter="bandpass", low=500, high=6000, normalize="noise", channel=sample.channel, db=db)

    c = spidb.Classification(
        datetime = sample.datetime,
        classifier = "nspa 500-6000",
        classification = nspa,
        sensor_id = sample.sensor.id,
        sample_id = sample.id,
        record_id = sample.record.id
    )
    db.session.add(c)
    db.session.commit()
    db.session.close()

db = spidb.Database(r"data/spi.db")

if __name__ == "__main__":

    db = spidb.Database(r"data/spi.db")

    sensor = db.session.get(spidb.Sensor, 1)

    if sensor is None:
        raise ValueError("Sensor with id 1 not found in the database.")

    for channel in sensor.channels: 
        channel.gain = normalization.noise_coefficient(db, sensor, channel, "bandpass", 500, 6000, order=10)
        db.session.commit()

    samples = db.session.query(spidb.Sample).filter(spidb.Sample.sensor == sensor).all()
    samples = [sample.id for sample in samples]

    cpus = mp.cpu_count()
    pool = mp.Pool(cpus)
    results = pool.map(generate_metrics, samples)
    pool.close()
    pool.join()