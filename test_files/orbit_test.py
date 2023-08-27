from sgp4.api import Satrec
from sgp4.api import jday

from astropy import units as u
from poliastro.bodies import Earth, Mars, Sun
from poliastro.twobody import Orbit

import numpy as np


#SGP4 TEST, TEME frame
s = '1 25544U 98067A   19343.69339541  .00001764  00000-0  38792-4 0  9991'
t = '2 25544  51.6439 211.2001 0007417  17.6667  85.6398 15.50103472202482'
satellite = Satrec.twoline2rv(s, t)


#jd, fr = jday(2023, 7, 15, 0, 0, 0) #convert date to Julian date
#print(jd,fr)
dfr = 0.5
jd, fr = 2458827, 0.0
e, r, v = satellite.sgp4(jd, fr)
#print(r)
#print(v)
#print(satellite.a)


#POLIASTRO TEST, pseudo-GCRS frame
#r = [3520.6039635075385, -2626.7656468119912, 5174.400087591946] << u.km
#v = [5.7242588849151295, 4.902309262806894, -1.3952863346878828] << u.km / u.s
r = r << u.km
v = v << u.km/u.s

orb = Orbit.from_vectors(Earth, r, v)
#print(orb.a/6378)

#propogation of each
for i in range(1):
    fr += dfr
    e, r, v = satellite.sgp4(jd, fr)
    orb = orb.propagate(12 << u.h)
    #print(orb.v.value)
