# Data card: exam conflict graph

## Shipped file (`exams.csv`)

A synthetic development instance of 8 courses and 13 conflicting pairs with
3 to 24 shared students per pair, built so that (i) the four largest-degree
courses form a clique, so no conflict-free 3-slot timetable exists and the
optimum has a positive cost; (ii) the greedy timetable is valid but
sub-optimal on the 6- and 8-course sub-instances (quality q between 0.75 and
0.80); and (iii) the optimum is unique up to relabelling of the slots.
It is intended for development, unit tests and pilot runs only.

## Departmental instance (to be added in week 3)

* Source: enrolment lists for 10 to 15 courses of one department, one semester.
* Processing: for every pair of courses the number of students enrolled in both
  is counted; pairs with no students in common are omitted.
* Fields: `course_a`, `course_b` (course codes only), `shared_students` (integer).
* Anonymisation: no student identifiers, names, grades or demographic fields are
  stored at any stage; counts below a threshold of 3 are rounded up to 3 so that
  no individual timetable can be inferred from a small count.
* Approval: written permission from the head of department is filed with the
  project before any data are copied.
* Licence: CC BY 4.0 for the derived counts; the enrolment lists are not released.

## Sub-instances

`src/qubo.sub_instance(G, k)` takes the k courses of largest weighted degree
(ties broken alphabetically), so sub-instances are nested and reproducible.
