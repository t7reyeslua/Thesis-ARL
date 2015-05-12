<?php
require_once 'connect.php';
/*$servername = "localhost";
$username = "root";
$password = "root";
$dbname = "plugwise";


$con = mysqli_connect($servername, $username, $password, $dbname);
*/
if (!$con)
{
die("Connection failed: " . mysqli_connect_error());
}


$plug_id = $_POST['name'];
$measurement_timestamp = $_POST['timestamp'];
$current_power = $_POST['power'];

//echo "coming";
$sql = "INSERT INTO log (plug_id, measurement_timestamp, current_power) VALUES ('$plug_id', '$measurement_timestamp', '$current_power');";

if (!mysqli_query($con,$sql))
{
die("Problem with query: " .  mysqli_error($con));
}

//echo "1 record added: " . $measurement_timestamp . " " . $appliance_name . " " . $current_power;

//mysqli_close($con);

?>
