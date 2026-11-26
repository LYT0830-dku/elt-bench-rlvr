SELECT
    customers.customer_id,
    customers.customer_name,
    COUNT(orders.order_id) AS total_orders,
    COALESCE(SUM(orders.amount), 0) AS net_sales
FROM {{ source('input', 'customers') }} AS customers
LEFT JOIN {{ source('input', 'orders') }} AS orders
    ON customers.customer_id = orders.customer_id
GROUP BY customers.customer_id, customers.customer_name
