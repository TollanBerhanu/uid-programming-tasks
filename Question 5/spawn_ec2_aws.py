"""
Provide Python scripts to programmatically spawn a VM on AWS.
The script should include parameters for accepting a user’s credentials and the\
type of the VM.
"""
from boto3 import client
from botocore.exceptions import ClientError

# Defina a function for spawning AWS EC2 (Elastic Compute Cloud) instance
def spawn_aws_ec2_instance(access_key, secret_key, instance_type, region, ami_id):
    # Establish connection to AWS EC2
    try:
        ec2 = client('ec2', aws_access_key_id=access_key,
                        aws_secret_access_key=secret_key,
                        region_name=region)
    except ClientError as err:
        print('Unable to create EC2 client: ', err)

    # Launch EC2 instance
    try:
        connection = ec2.run_instances(
            ImageId=ami_id,
            InstanceType=instance_type,
            MaxCount=1,
            MinCount=1
        )

        # instance_id = connection['Instances'][0]['InstanceId']
        return connection
    except ClientError as err:
        print('Unable to run EC2 instance: ', err)


if __name__ == '__main__':
    # Accepting user credentials and instance type
    access_key = input('* AWS Access Key ID: ')
    secret_key = input('* AWS Secret Access Key: ')
    # Set the limted free instance as default
    instance_type = input('EC2 Instance Type: ') or 't2.micro'
    # Set default region to Eastern US
    region = input('Enter AWS Region: ') or 'us-east-1'
    # Set default Image id to "Ubuntu Server 18.04 LTS (HVM), SSD Volume Type (ami-02d55cb47e83a99a0)"
    ami_id = input('Enter AMI ID: ') or 'ami-02d55cb47e83a99a0'
    
    # Spawn AWS EC2 instance
    connection = spawn_aws_ec2_instance(access_key=access_key,
                                        secret_key=secret_key,
                                        instance_type=instance_type,
                                        region=region,
                                        ami_id=ami_id)
    if connection:
        print('AWS EC2 instance spawned successfully!')
    else:
        print('Unable to spawn AWS EC2 instance!')
