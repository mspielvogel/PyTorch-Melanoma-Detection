package org.pytorch.demo;

import android.content.Context;
import android.view.LayoutInflater;
import android.view.View;
import android.widget.TextView;

import androidx.annotation.Nullable;

public class InfoViewFactory {
  public static final int INFO_VIEW_TYPE_IMAGE_CLASSIFICATION_RESNET = 1;
  public static final int INFO_VIEW_TYPE_IMAGE_CLASSIFICATION_QMOBILENET = 2;
  public static final int INFO_VIEW_TYPE_TEXT_CLASSIFICATION = 3;

  public static View newInfoView(Context context, int infoViewType, @Nullable String additionalText) {
    LayoutInflater inflater = LayoutInflater.from(context);
    if (INFO_VIEW_TYPE_IMAGE_CLASSIFICATION_RESNET == infoViewType) {
      View view = inflater.inflate(R.layout.info, null, false);
      TextView infoTextView = view.findViewById(R.id.info_title);
      TextView descriptionTextView = view.findViewById(R.id.info_description);

      infoTextView.setText(R.string.vision_card_resnet_title);
      StringBuilder sb = new StringBuilder(context.getString(R.string.vision_card_resnet_description));
      if (additionalText != null) {
        sb.append('\n').append(additionalText);
      }
      descriptionTextView.setText(sb.toString());
      return view;
    } else if (INFO_VIEW_TYPE_IMAGE_CLASSIFICATION_QMOBILENET == infoViewType) {
      View view = inflater.inflate(R.layout.info, null, false);
      TextView infoTextView = view.findViewById(R.id.info_title);
      TextView descriptionTextView = view.findViewById(R.id.info_description);

      infoTextView.setText(R.string.vision_card_qmobilenet_title);
      StringBuilder sb = new StringBuilder(context.getString(R.string.vision_card_qmobilenet_description));
      if (additionalText != null) {
        sb.append('\n').append(additionalText);
      }
      descriptionTextView.setText(sb.toString());
      return view;
    } else if (INFO_VIEW_TYPE_TEXT_CLASSIFICATION == infoViewType) {
      View view = inflater.inflate(R.layout.info, null, false);
      TextView infoTextView = view.findViewById(R.id.info_title);
      TextView descriptionTextView = view.findViewById(R.id.info_description);

      return view;
    }
    throw new IllegalArgumentException("Unknown info view type");
  }

  public static View newErrorDialogView(Context context) {
    LayoutInflater inflater = LayoutInflater.from(context);
    View view = inflater.inflate(R.layout.error_dialog, null, false);
    return view;
  }
}
