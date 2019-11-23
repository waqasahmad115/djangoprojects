# Generated by Django 2.2.6 on 2019-11-23 08:21

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('POI_Record', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='MyCamera',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('CCID', models.IntegerField(unique=True)),
                ('Logitude', models.DecimalField(decimal_places=2, max_digits=12)),
                ('Latitude', models.DecimalField(decimal_places=2, max_digits=12)),
            ],
            options={
                'db_table': 'camera_mycamera',
            },
        ),
        migrations.CreateModel(
            name='MyCity',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('CID', models.IntegerField(unique=True)),
                ('city_name', models.CharField(max_length=255)),
            ],
            options={
                'db_table': 'camera_mycity',
            },
        ),
        migrations.CreateModel(
            name='MyZone',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('ZID', models.IntegerField(unique=True)),
                ('Zone_area_name', models.CharField(max_length=255)),
                ('radius', models.DecimalField(decimal_places=2, max_digits=12)),
                ('Latitude', models.DecimalField(decimal_places=2, max_digits=12)),
                ('Logitude', models.DecimalField(decimal_places=2, max_digits=12)),
                ('CID', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='camera.MyCity')),
            ],
            options={
                'db_table': 'camera_myzone',
            },
        ),
        migrations.CreateModel(
            name='MyDetected_Poi',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('DID', models.IntegerField(unique=True)),
                ('detected_image', models.ImageField(upload_to='')),
                ('date_time', models.DateField()),
                ('cameraID', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='camera.MyCamera')),
                ('poiID', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='POI_Record.MyPoiRecord')),
            ],
            options={
                'db_table': 'camera_mydetected_poi',
            },
        ),
        migrations.AddField(
            model_name='mycamera',
            name='ZoneID',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='camera.MyZone'),
        ),
    ]
