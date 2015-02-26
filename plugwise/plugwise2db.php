<?php
$servername = "localhost";
$username = "root";
$password = "tudelftlws336";
$dbname = "thesisdb";


$con = mysqli_connect($servername, $username, $password, $dbname);

if (!$con)
{
die("Connection failed: " . mysqli_connect_error());
}

$appliance_name = $_POST['name'];
$measurement_timestamp = $_POST['timestamp'];
$current_power = $_POST['power'];

$sql = "INSERT INTO tbl_appliances_power (name, measurement_timestamp, current_power) VALUES ('$appliance_name', '$measurement_timestamp', '$current_power');";

if (!mysqli_query($con,$sql))
{
die("Problem with query: " .  mysqli_error($con));
}

echo "1 record added: " . $measurement_timestamp . " " . $appliance_name . " " . $current_power;

mysqli_close($con);

?>