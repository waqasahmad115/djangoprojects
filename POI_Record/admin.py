from django.contrib import admin
from .models import MyPoiRecord
from django.contrib.admin.options import ModelAdmin 
from inline_actions.admin import InlineActionsMixin
from inline_actions.admin import InlineActionsModelAdminMixin


class MyPoiRecordAdmin(ModelAdmin):
    list_display=["name","age","comments","threat_level","image1","image2","image3","image4","image5","image6","image7","image8","image9","image10"]
    search_fields=["name"]
    list_filter=["name"]
    actions = ['cancel_orders', ]
#     def embedding_actions(self, obj):
#             # TODO: Render action buttons
#     def get_urls(self):
#         urls = super().get_urls()
#         custom_urls = [
#             url(
#                 r'^(?P<account_id>.+)/deposit/$',
#                 self.admin_site.admin_view(self.process_deposit),
#                 name='account-deposit',
#             ),
#             url(
#                 r'^(?P<account_id>.+)/withdraw/$',
#                 self.admin_site.admin_view(self.process_withdraw),
#                 name='account-withdraw',
#             ),
#         ]
#         return custom_urls + urls
#     def account_actions(self, obj):
#         return format_html(
#             '<a class="button" href="{}">Deposit</a>&nbsp;'
#             '<a class="button" href="{}">Withdraw</a>',
#             reverse('admin:account-deposit', args=[obj.pk]),
#             reverse('admin:account-withdraw', args=[obj.pk]),
#         )
#     account_actions.short_description = 'Account Actions'
#     account_actions.allow_tags = True
# class MyPoiRecordInline(InlineActionsMixin,
#                     admin.TabularInline):
#     model = MyPoiRecord
#     inline_actions = ('view',)

#     def has_add_permission(self):
#         return False
    

#     def view(self, request, obj, parent_obj=None):
#         url = reverse(
#             'admin:{}_{}_change'.format(
#                 obj._meta.app_label,
#                 obj._meta.model_name,
#             ),
#             args=(obj.pk,)
#         )
#         return redirect(url)
   #view.short_description = _("View")

admin.site.register(MyPoiRecord,MyPoiRecordAdmin)
class MyPoiRecordAdmin(admin.ModelAdmin):
    change_list_template = "POI_Record/embeddings.html"

    def actions_html(self, obj):
        return format_html('<button class="btn" type="button" onclick="activate_and_send_email({pk})">Activate and send email</button>', pk=obj.pk)

    actions_html.allow_tags = True
    actions_html.short_description = "Actions"







from django.utils.html import format_html
class MyPoiRecordAdmin(admin.ModelAdmin):
    
    def get_urls(self):
        urls = super().get_urls()
        # custom_urls = [
        #     url(
        #         r'^(?P<account_id>.+)/deposit/$',
        #         self.admin_site.admin_view(self.process_deposit),
        #         name='account-deposit',
        #     ),
        #     url(
        #         r'^(?P<account_id>.+)/withdraw/$',
        #         self.admin_site.admin_view(self.process_withdraw),
        #         name='account-withdraw',
        #     ),
        # ]
        # return custom_urls + urls