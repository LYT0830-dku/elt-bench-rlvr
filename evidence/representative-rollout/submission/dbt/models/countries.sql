with cars as (
    select distinct
        p.country as country_id,
        d.id as id,
        d.car_name as car_name,
        d.mpg as mpg,
        d.acceleration as acceleration,
        pr.price as price
    from {{ source('airbyte_schema', 'production') }} p
    inner join {{ source('airbyte_schema', 'data') }} d on p.id = d.id
    left join {{ source('airbyte_schema', 'price') }} pr on d.id = pr.id
),
ranked as (
    select
        c.*,
        row_number() over (partition by c.country_id order by c.acceleration asc, c.car_name asc) as fastest_rank,
        row_number() over (partition by c.country_id order by c.mpg desc, c.car_name desc) as eff_rank
    from cars c
)
select
    co.origin as origin,
    co.country as country,
    max(case when r.fastest_rank = 1 then r.car_name end) as fastest_car,
    max(case when r.eff_rank = 1 then r.car_name end) as most_fuel_efficient_car,
    coalesce(count(distinct r.id), 0) as num_cars,
    coalesce(sum(case when r.price > 35000 and r.acceleration = 14 then 1 else 0 end), 0) as num_car_worth_over_35000_and_acceleration_14,
    avg(r.price) as avg_car_price
from {{ source('airbyte_schema', 'country') }} co
left join ranked r on co.origin = r.country_id
group by co.origin, co.country
